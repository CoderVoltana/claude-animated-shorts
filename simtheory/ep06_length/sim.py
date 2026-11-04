"""SimTheory #06 - Does length change everything?
Physics + measurements. Run first; everything else renders from data/sim.npz + data/measurements.json.

Model: planar double pendulum, point masses m1=m2=1 kg on massless rigid rods L1 (upper), L2 (lower),
angles measured from the downward vertical (counter-clockwise positive), x = L sin(th), y = -L cos(th).
Integrator: classical RK4, float64, fixed step dt.
"""
import json, math, time
import numpy as np
from numba import njit

G = 9.81
M1 = M2 = 1.0
RATIOS = [  # label, L1, L2, colour
    ("1:3", 0.5, 1.5, "#22E5FF"),
    ("1:1", 1.0, 1.0, "#3D7BFF"),
    ("3:1", 1.5, 0.5, "#9B5CFF"),
    ("9:1", 1.8, 0.2, "#FF8A2A"),
]
TH0 = math.pi / 2          # 90 deg both arms
EPS = 1e-9                 # twin offset in th1 (rad)
SEP = math.radians(1.0)    # chaos-clock threshold on bottom-arm angle
T_CHAOS = 60.0             # chaos clock horizon (s sim)
T_SAVE = 30.0              # saved trajectory length (video uses 0..28.5 s)
SAVE_HZ = 2400             # saved sample rate (40 samples per 60 fps frame)
PX_PER_M_MAX = 260.0       # largest on-screen scale used in the video (overlay art) -> "visible" threshold
VISIBLE_PX = 0.5           # half a pixel at the largest scale = not visible


@njit(cache=True)
def deriv(s, L1, L2):
    t1, t2, w1, w2 = s[0], s[1], s[2], s[3]
    d = t1 - t2
    den = 2 * M1 + M2 - M2 * math.cos(2 * t1 - 2 * t2)
    a1 = (-G * (2 * M1 + M2) * math.sin(t1) - M2 * G * math.sin(t1 - 2 * t2)
          - 2 * math.sin(d) * M2 * (w2 * w2 * L2 + w1 * w1 * L1 * math.cos(d))) / (L1 * den)
    a2 = (2 * math.sin(d) * (w1 * w1 * L1 * (M1 + M2) + G * (M1 + M2) * math.cos(t1)
          + w2 * w2 * L2 * M2 * math.cos(d))) / (L2 * den)
    return np.array((w1, w2, a1, a2))


@njit(cache=True)
def rk4(s, dt, L1, L2):
    k1 = deriv(s, L1, L2)
    k2 = deriv(s + 0.5 * dt * k1, L1, L2)
    k3 = deriv(s + 0.5 * dt * k2, L1, L2)
    k4 = deriv(s + dt * k3, L1, L2)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def energy(s, L1, L2):
    # potential measured from the lowest configuration (both bobs hanging straight down)
    t1, t2, w1, w2 = s[0], s[1], s[2], s[3]
    y1 = -L1 * math.cos(t1)
    y2 = y1 - L2 * math.cos(t2)
    v1sq = (L1 * w1) ** 2
    v2sq = v1sq + (L2 * w2) ** 2 + 2 * L1 * L2 * w1 * w2 * math.cos(t1 - t2)
    T = 0.5 * M1 * v1sq + 0.5 * M2 * v2sq
    V = M1 * G * (y1 + L1) + M2 * G * (y2 + L1 + L2)
    return T + V


@njit(cache=True)
def run(s0, dt, n_steps, save_every, L1, L2):
    n_save = n_steps // save_every + 1
    out = np.empty((n_save, 4))
    en = np.empty(n_save)
    s = s0.copy()
    out[0] = s
    en[0] = energy(s, L1, L2)
    j = 1
    for i in range(1, n_steps + 1):
        s = rk4(s, dt, L1, L2)
        if i % save_every == 0:
            out[j] = s
            en[j] = energy(s, L1, L2)
            j += 1
    return out, en


@njit(cache=True)
def chaos_clock(s0a, s0b, dt, n_steps, L1, L2, sep):
    a = s0a.copy()
    b = s0b.copy()
    for i in range(1, n_steps + 1):
        a = rk4(a, dt, L1, L2)
        b = rk4(b, dt, L1, L2)
        d = a[1] - b[1]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > sep:
            return i * dt
    return -1.0


def count_flips(phi):
    """Flip = the bottom arm passes straight up (absolute angle crosses an odd multiple of pi) AND then swings
    down past horizontal on the far side (comes within 90 deg of the next straight-down direction) before
    crossing back. A wobble over the top that falls back the way it came is not a flip. The tick is stamped
    at the sample where the arm crossed the top."""
    k = np.floor((phi + np.pi) / (2 * np.pi)).astype(np.int64)
    R = int(k[0]); cand = None
    events = []
    for i in range(1, len(phi)):
        if k[i] != k[i - 1]:
            if k[i] == R:
                cand = None                       # fell back over the same top: discard
            else:
                cand = (i, int(k[i]))
        if cand is not None and abs(phi[i] - 2 * np.pi * cand[1]) <= np.pi / 2:
            events.append(cand[0]); R = cand[1]; cand = None
    cum = np.zeros(len(phi), np.int64)
    for e in events:
        cum[e:] += 1
    return cum, np.array(events) / SAVE_HZ


def bob2(traj, L1, L2):
    x = L1 * np.sin(traj[:, 0]) + L2 * np.sin(traj[:, 1])
    y = -L1 * np.cos(traj[:, 0]) - L2 * np.cos(traj[:, 1])
    return x, y


def main():
    t0 = time.time()
    s0 = np.array([TH0, TH0, 0.0, 0.0])
    log = {"model": "double pendulum, m1=m2=1 kg, g=9.81, no friction, RK4 float64",
           "release": "theta1=theta2=90 deg from rest", "twin_offset_rad": EPS,
           "chaos_threshold_deg": 1.0, "chaos_horizon_s": T_CHAOS}

    # ---- 1. timestep selection: halve dt until halving no longer visibly changes the first 10 s
    cands = [1 / 2400, 1 / 4800, 1 / 9600, 1 / 19200]
    halving = []
    chosen = None
    for dt in cands:
        row = {"dt_s": dt, "pendulums": {}}
        ok_all = True
        for lab, L1, L2, _ in RATIOS:
            n = int(round(10.0 / dt))
            A, _ = run(s0, dt, n, 1, L1, L2)
            B, _ = run(s0, dt / 2, 2 * n, 2, L1, L2)
            xa, ya = bob2(A, L1, L2)
            xb, yb = bob2(B, L1, L2)
            dd = np.hypot(xa - xb, ya - yb)
            err_m = float(np.max(dd))
            err_px = err_m * PX_PER_M_MAX
            ok = err_px < VISIBLE_PX
            ok_all &= ok
            row["pendulums"][lab] = {"max_bob2_diff_m_first10s": err_m, "max_diff_px_at_260px_per_m": err_px, "pass": bool(ok)}
        for lab, L1, L2, _ in RATIOS:
            n = int(round(28.5 / dt))
            A, _ = run(s0, dt, n, 1, L1, L2)
            B, _ = run(s0, dt / 2, 2 * n, 2, L1, L2)
            xa, ya = bob2(A, L1, L2); xb, yb = bob2(B, L1, L2)
            bad = np.nonzero(np.hypot(xa - xb, ya - yb) * PX_PER_M_MAX >= VISIBLE_PX)[0]
            row["pendulums"][lab]["identical_to_dt_half_until_s"] = float(bad[0] * dt) if len(bad) else 28.5
        row["pass_all"] = bool(ok_all)
        halving.append(row)
        print(f"dt={dt:.3e}", {k: f"{v['max_diff_px_at_260px_per_m']:.2e}px" for k, v in row["pendulums"].items()}, ok_all)
        if ok_all:
            chosen = dt
            break
    assert chosen is not None
    dt = chosen
    log["dt_s"] = dt
    log["halving_check"] = halving
    log["halving_check_rule"] = (f"max distance between bottom-bob paths at dt and dt/2 over the first 10 s "
                                 f"< {VISIBLE_PX} px at the largest on-screen scale ({PX_PER_M_MAX:.0f} px/m)")

    # ---- 2. main trajectories (saved), energy over 60 s, chaos clock with twin
    save_every = int(round((1 / SAVE_HZ) / dt))
    assert abs(save_every * dt - 1 / SAVE_HZ) < 1e-15
    n_save_steps = int(round(T_SAVE / dt))
    n_chaos = int(round(T_CHAOS / dt))
    trajs, res = [], {}
    for lab, L1, L2, col in RATIOS:
        tr, _ = run(s0, dt, n_save_steps, save_every, L1, L2)
        _, en60 = run(s0, dt, n_chaos, save_every, L1, L2)
        E0 = en60[0]
        drift = float(np.max(np.abs(en60 - E0)) / E0)
        s0b = s0.copy(); s0b[0] += EPS
        tc = chaos_clock(s0, s0b, dt, n_chaos, L1, L2, SEP)
        tc_half = chaos_clock(s0, s0b, dt / 2, 2 * n_chaos, L1, L2, SEP)
        # flips: bottom arm passes through straight-up (absolute th2 crosses an odd multiple of pi)
        flips_cum, flip_t = count_flips(tr[:, 1])
        trajs.append(tr)
        x2, y2 = bob2(tr, L1, L2)
        y1 = -L1 * np.cos(tr[:, 0])
        res[lab] = {
            "L1_m": L1, "L2_m": L2, "colour": col,
            "E0_J": float(E0), "max_energy_drift_rel_60s": drift, "energy_pass": drift < 1e-3,
            "chaos_clock_s": None if tc < 0 else float(tc),
            "chaos_clock_s_at_dt_half": None if tc_half < 0 else float(tc_half),
            "flips_cum": flips_cum,
            "flip_times_sim_s": [round(float(x), 4) for x in flip_t],
            "flips_at_sim_16_5s": int(flips_cum[int(16.5 * SAVE_HZ)]),
            "flips_at_sim_28_5s_final_shown": int(flips_cum[int(28.5 * SAVE_HZ)]),
            "max_bob2_height_above_pivot_m": float(max(y2.max(), 0)),
            "max_bob1_height_above_pivot_m": float(max(y1.max(), 0)),
            "max_bob2_speed_mps": float(np.max(np.hypot(np.gradient(x2), np.gradient(y2)) * SAVE_HZ)),
        }
        print(lab, "E0", E0, "drift", drift, "chaos", tc, tc_half, "flips(28.5s)", flips_cum[int(28.5 * SAVE_HZ)])
    np.savez_compressed("data/sim.npz", traj=np.stack(trajs), save_hz=SAVE_HZ, dt=dt,
                        flips=np.stack([res[l]["flips_cum"] for l, *_ in RATIOS]),
                        L=np.array([[a, b] for _, a, b, _ in RATIOS]))
    for lab in res:
        del res[lab]["flips_cum"]
    log["pendulums"] = res
    log["sim_seconds"] = time.time() - t0
    json.dump(log, open("data/physics.json", "w"), indent=2)
    print("done in", log["sim_seconds"])


if __name__ == "__main__":
    main()
