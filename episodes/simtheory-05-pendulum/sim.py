"""Step 1: run the full physics simulation and all measurements, save every state to disk.

Numeric pendulums: full nonlinear theta'' = -(g/L) sin(theta), classic RK4, float64, dt = 1/4800 s.
Threshold pendulum (E = E_c): exact analytic separatrix solution
    delta(t) = 4 atan(exp(-w0 t)),  theta = pi - delta,  omega = 2 w0 sech(w0 t)
Outputs:
    data/sim.npz            every state of every pendulum (theta, omega at every RK4 step)
    data/measurements.json  all checks and measured numbers
"""
import json
import math
import os

import numpy as np

from config import DT, EC, FRACS, G, L, M, NUM, PUSH, T_SIM, THR, W0, WC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def rk4_run(w_init, dt, n):
    """Vectorised RK4 for several independent pendulums. Returns theta, omega arrays of shape (n+1, k)."""
    k = len(w_init)
    th = np.zeros((n + 1, k))
    om = np.zeros((n + 1, k))
    th[0] = 0.0
    om[0] = np.asarray(w_init, float)
    a = -(G / L)
    x, v = th[0].copy(), om[0].copy()
    h2, h6 = dt / 2.0, dt / 6.0
    for s in range(n):
        k1x, k1v = v, a * np.sin(x)
        x2, v2 = x + h2 * k1x, v + h2 * k1v
        k2x, k2v = v2, a * np.sin(x2)
        x3, v3 = x + h2 * k2x, v + h2 * k2v
        k3x, k3v = v3, a * np.sin(x3)
        x4, v4 = x + dt * k3x, v + dt * k3v
        k4x, k4v = v4, a * np.sin(x4)
        x = x + h6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + h6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        th[s + 1], om[s + 1] = x, v
    return th, om


def energy(th, om):
    return 0.5 * M * L * L * om ** 2 + M * G * L * (1.0 - np.cos(th))


def gap_rad(t):
    """Gap to the top, computed DIRECTLY (never as pi - theta). Accurate down to ~1e-300."""
    return 4.0 * np.arctan(np.exp(-W0 * np.asarray(t, float)))


def threshold_state(t):
    t = np.asarray(t, float)
    d = gap_rad(t)
    return math.pi - d, 2.0 * W0 / np.cosh(W0 * t), d


def bottom_crossings(t, th):
    """Times at which theta passes a multiple of 2*pi (the bottom), linear interpolation inside a step."""
    k = np.floor((th + math.pi) / (2 * math.pi))          # which 'bottom' cell we are in
    out = []
    # crossing of theta = 2 pi n: sign change of sin(theta/2)*... simpler: track wrapped value through 0
    wr = (th + math.pi) % (2 * math.pi) - math.pi
    s = np.sign(wr)
    idx = np.where((s[:-1] <= 0) & (s[1:] > 0) | (s[:-1] >= 0) & (s[1:] < 0))[0]
    for i in idx:
        if abs(wr[i + 1] - wr[i]) > math.pi:              # wrap at the top, not a bottom pass
            continue
        a, b = wr[i], wr[i + 1]
        if a == b:
            continue
        f = a / (a - b)
        out.append(t[i] + f * (t[i + 1] - t[i]))
    # drop duplicates from exact-zero samples
    res = []
    for x in out:
        if not res or x - res[-1] > 1e-6:
            res.append(x)
    return np.array(res)


def top_crossings(t, th):
    """Times at which theta passes an odd multiple of pi (over the top)."""
    u = th - math.pi
    wr = (u + math.pi) % (2 * math.pi) - math.pi
    res = []
    for i in range(len(t) - 1):
        a, b = wr[i], wr[i + 1]
        if abs(b - a) > math.pi or (a < 0) == (b < 0) or a == b:
            continue
        res.append(t[i] + a / (a - b) * (t[i + 1] - t[i]))
    return np.array(res)


def main():
    os.makedirs(DATA, exist_ok=True)
    n = int(round(T_SIM / DT))
    t = np.arange(n + 1) * DT
    w_init = [PUSH[i] for i in NUM]
    th, om = rk4_run(w_init, DT, n)

    # analytic threshold pendulum, on the same grid (for convenience; renderer evaluates it analytically anyway)
    th_thr, om_thr, d_thr = threshold_state(t)
    TH = np.zeros((n + 1, len(FRACS)))
    OM = np.zeros((n + 1, len(FRACS)))
    for j, i in enumerate(NUM):
        TH[:, i], OM[:, i] = th[:, j], om[:, j]
    TH[:, THR], OM[:, THR] = th_thr, om_thr

    meas = {"dt_s": DT, "steps": n, "t_end_s": T_SIM, "integrator": "RK4, float64",
            "omega0": W0, "omega_c": WC, "E_c_J": EC, "pendulums": []}

    ok_all = True
    for i, f in enumerate(FRACS):
        rec = {"fraction": f, "push_speed_rad_s": PUSH[i], "E0_J": f * EC}
        if i == THR:
            rec["method"] = "exact analytic separatrix solution"
            rec["energy_rel_drift_max"] = 0.0
            rec["phase_curve_max_err"] = 0.0
            meas["pendulums"].append(rec)
            continue
        E = energy(TH[:, i], OM[:, i])
        E0 = E[0]
        drift = float(np.max(np.abs(E - E0)) / E0)
        # phase-curve check: simulated |omega| vs exact energy curve, relative to the curve's peak speed
        w2 = 2 * E0 / (M * L * L) - 2 * W0 ** 2 * (1 - np.cos(TH[:, i]))
        w_ex = np.sqrt(np.maximum(w2, 0.0))
        err = float(np.max(np.abs(np.abs(OM[:, i]) - w_ex)) / PUSH[i])
        bot = bottom_crossings(t, TH[:, i])
        top = top_crossings(t, TH[:, i])
        rec.update(method="RK4 simulation", energy_rel_drift_max=drift, phase_curve_max_err=err,
                   theta_max_deg=float(np.degrees(np.max(np.abs(TH[:, i])))) if f < 1 else None,
                   min_omega=float(np.min(OM[:, i])),
                   flips_over_top=int(len(top)), bottom_passes=int(len(bot)))
        if f < 1:
            # analytic amplitude: cos(theta_max) = 1 - 2f
            rec["theta_max_exact_deg"] = math.degrees(math.acos(1 - 2 * f))
        rec["pass_drift"] = drift < 1e-4
        rec["pass_phase"] = err < 5e-3
        ok_all &= rec["pass_drift"] and rec["pass_phase"]
        meas["pendulums"].append(rec)

    # check 3: 0.95 never passes the top, 1.05 always does
    i95, i105 = FRACS.index(0.95), FRACS.index(1.05)
    never = bool(np.max(np.abs(TH[:, i95])) < math.pi)
    always = bool(np.min(OM[:, i105]) > 0)        # never turns back -> passes the top every revolution
    flips105 = int(len(top_crossings(t, TH[:, i105])))
    meas["check_095_never_flipped"] = never
    meas["check_095_max_angle_deg"] = float(np.degrees(np.max(np.abs(TH[:, i95]))))
    meas["check_105_always_flipped"] = always
    meas["check_105_min_omega"] = float(np.min(OM[:, i105]))
    meas["check_105_flips_in_40s"] = flips105
    ok_all &= never and always

    # gap spot checks (computed directly from delta)
    def deg(x):
        return float(np.degrees(x))
    meas["gap_deg"] = {str(tt): deg(gap_rad(tt)) for tt in (1.0, 3.0, 5.0)}
    # analytic solution residual: theta'' + w0^2 sin(theta) = 0, checked with exact derivative
    tt = np.linspace(0, 8, 2001)
    # theta'' = d/dt (2 w0 sech(w0 t)) = -2 w0^2 sech tanh ; sin(theta) = sin(delta)
    thdd = -2 * W0 ** 2 / np.cosh(W0 * tt) * np.tanh(W0 * tt)
    res = thdd + W0 ** 2 * np.sin(gap_rad(tt))
    meas["analytic_ode_residual_max"] = float(np.max(np.abs(res)))
    # analytic energy: should equal E_c exactly
    th_a, om_a, d_a = threshold_state(tt)
    Ea = 0.5 * om_a ** 2 + G * (1 + np.cos(d_a))        # 1 - cos(pi - d) = 1 + cos d
    meas["analytic_energy_rel_err_max"] = float(np.max(np.abs(Ea - EC) / EC))

    # demonstration: a *numerical* threshold pendulum falls off the knife-edge
    thn, omn = rk4_run([WC], DT, n)
    d_num = math.pi - thn[:, 0]
    ii = np.where(np.abs(thn[:, 0] - math.pi) > 0.05)[0]
    ii = ii[ii > int(3 / DT)]
    meas["numeric_threshold_leaves_top_at_s"] = float(t[ii[0]]) if len(ii) else None
    meas["numeric_threshold_fate"] = (None if not len(ii) else
                                      ("flipped over" if thn[ii[0], 0] > math.pi else "fell back"))
    meas["all_checks_pass"] = bool(ok_all)

    # bottom passes (for audio pings) on the sim clock
    pings = {}
    for i in NUM:
        pings[str(i)] = bottom_crossings(t, TH[:, i]).tolist()
    pings[str(THR)] = [0.0]
    tops = {str(i): top_crossings(t, TH[:, i]).tolist() for i in NUM if FRACS[i] > 1}

    np.savez_compressed(os.path.join(DATA, "sim.npz"), t=t, theta=TH, omega=OM, fracs=np.array(FRACS), dt=DT)
    json.dump({"bottom": pings, "top": tops}, open(os.path.join(DATA, "events_sim.json"), "w"))
    json.dump(meas, open(os.path.join(DATA, "measurements.json"), "w"), indent=2)

    print(f"steps={n}  dt={DT*1e3:.4f} ms")
    for r in meas["pendulums"]:
        print(f"  f={r['fraction']:<5} push={r['push_speed_rad_s']:.4f} rad/s  drift={r['energy_rel_drift_max']:.2e}"
              f"  phase_err={r['phase_curve_max_err']:.2e}  flips={r.get('flips_over_top')}")
    print("0.95 never flipped:", never, " max angle", meas["check_095_max_angle_deg"])
    print("1.05 always flipped:", always, " min omega", meas["check_105_min_omega"], "flips", flips105)
    print("gap deg:", meas["gap_deg"])
    print("analytic residual", meas["analytic_ode_residual_max"], " energy err", meas["analytic_energy_rel_err_max"])
    print("numeric threshold leaves top at", meas["numeric_threshold_leaves_top_at_s"], meas["numeric_threshold_fate"])
    print("ALL CHECKS PASS:", ok_all)
    if not ok_all:
        raise SystemExit("physics checks failed")


if __name__ == "__main__":
    main()
