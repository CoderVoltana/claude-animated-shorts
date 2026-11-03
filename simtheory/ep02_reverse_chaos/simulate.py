"""Run the real simulation, the checks, and store everything the renderer and the audio need (sim.npz, sim.json)."""
import json, math, time
import numpy as np
from physics import *
from config import *

DEG = math.pi / 180


def checks():
    out = {}
    # 1) Hamiltonian form == textbook Lagrangian (omega) form, at random states
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        t1, t2 = rng.uniform(-math.pi, math.pi, 2)
        w1, w2 = rng.uniform(-8, 8, 2)
        p1, p2 = momenta(t1, t2, w1, w2)
        # omega-dot from the Hamiltonian flow by tiny central differences of the exact flow map
        h = 1e-6
        a = []
        for sgn in (+1, -1):
            y = np.array([t1, t2, p1, p2])
            y2, _ = step(y, sgn * h)
            a.append(omegas(*y2))
        acc = ((a[0][0] - a[1][0]) / (2 * h), (a[0][1] - a[1][1]) / (2 * h))
        ref = lagrange_accel(t1, t2, w1, w2)
        worst = max(worst, abs(acc[0] - ref[0]) / (1 + abs(ref[0])), abs(acc[1] - ref[1]) / (1 + abs(ref[1])))
    out["hamilton_vs_lagrange_max_rel_err"] = worst
    # 2) small-angle, non-chaotic reversal sanity test: 10 deg / 5 deg, 5 s forward, reverse, 5 s forward
    n = int(round(5.0 / DT))
    y0 = start_state(10.0, 5.0)
    F, _ = integrate(y0, DT, n)
    B, _ = integrate(reverse(F[-1]), DT, n)
    err = float(np.max(np.abs(B[-1, :2] - y0[:2])))
    out["sanity_small_angle_return_err_rad"] = err
    out["sanity_small_angle_pass"] = bool(err < 1e-9)
    return out


def main():
    t0 = time.time()
    N = int(round(T_SIM / DT))
    y0 = start_state(TH0_DEG, TH0_DEG)
    F, it_f = integrate(y0, DT, N)                    # forward run, every state recorded
    yr = reverse(F[N])                                # omega -> -omega (exact)
    C, it_c = integrate(yr, DT, N)                    # CYAN: rounding only
    yo = yr.copy()
    yo[0] = yo[0] + NUDGE_DEG * DEG                   # ORANGE: theta1 nudged by 1e-10 deg
    O, it_o = integrate(yo, DT, N)
    Gh = F[::-1].copy()                               # GHOST: recorded run played backwards
    print("integrated", f"{time.time()-t0:.1f}s", "max fixed-point iterations", it_f, it_c, it_o)

    E0 = energy(*y0)
    EF = energies(F); EC = energies(C); EO = energies(O)
    drift = float(max(np.max(np.abs(EF - E0)), np.max(np.abs(EC - E0)), np.max(np.abs(EO - E0))) / abs(E0))

    tau = np.arange(N + 1) * DT                       # time since the reversal
    res = {}
    for name, X in (("cyan", C), ("orange", O)):
        dth = np.abs(wrap(X[:, :2] - Gh[:, :2]))
        ang = np.max(dth, axis=1)                     # rad
        xa, ya, xb, yb = bobs(X[:, 0], X[:, 1])
        xg, yg, xh, yh = bobs(Gh[:, 0], Gh[:, 1])
        dist = np.hypot(xb - xh, yb - yh)            # m, lower bobs
        k = int(np.argmax(ang > DIVERGE_DEG * DEG))
        res[name] = dict(ang=ang, dist=dist, k=k)
        print(name, "diverges (>2 deg) at tau =", k * DT, f"({k*DT/T_SIM:.1%} of the way back)")

    # Lyapunov exponent: least-squares slope of ln(angle separation) vs tau (raw per-step data, every 10 ms).
    # (a) COMMON window for both copies: tau in [0.5 s, tau where ORANGE first reaches 1e-3 rad] (pre-saturation)
    # (b) each copy's own window: first separation >= 1e-15 rad (cyan) / start (orange) -> first reach of 1e-3 rad
    def fit(a, k0, k1):
        sel = np.arange(k0, k1, 100)
        sel = sel[a[sel] > 0]
        A = np.vstack([tau[sel], np.ones(len(sel))]).T
        sl, ic = np.linalg.lstsq(A, np.log(a[sel]), rcond=None)[0]
        r = np.corrcoef(tau[sel], np.log(a[sel]))[0, 1]
        return dict(slope=float(sl), intercept=float(ic), t_lo=float(tau[k0]), t_hi=float(tau[k1]), r2=float(r * r))
    ao, ac = res["orange"]["ang"], res["cyan"]["ang"]
    k_end = int(np.argmax(ao > 1e-3))
    k_beg = int(round(0.5 / DT))
    fits = {"common_cyan": fit(ac, k_beg, k_end), "common_orange": fit(ao, k_beg, k_end),
            "own_cyan": fit(ac, int(np.argmax(ac >= 1e-15)), int(np.argmax(ac > 1e-3))),
            "own_orange": fit(ao, 0, k_end)}
    fits["lambda_mean_common"] = 0.5 * (fits["common_cyan"]["slope"] + fits["common_orange"]["slope"])
    for kk, v in fits.items():
        print("fit", kk, v)
    for name in ("cyan", "orange"):
        res[name]["env"] = res[name]["ang"]
    chk = checks()
    print(chk)
    info = dict(TH0_DEG=TH0_DEG, T_SIM=T_SIM, DT=DT, steps=N, integrator="implicit midpoint (canonical th,p), fixed-point to machine precision",
                max_fixed_point_iters=int(max(it_f, it_c, it_o)), E0=E0, energy_drift_rel=drift,
                nudge_rad=NUDGE_DEG * DEG, theta1_rev=float(yr[0]), theta1_nudged=float(yo[0]),
                cyan_first_sep=float(res["cyan"]["ang"][1]), orange_first_sep=float(res["orange"]["ang"][0]),
                tau_orange=res["orange"]["k"] * DT, tau_cyan=res["cyan"]["k"] * DT,
                frac_orange=res["orange"]["k"] * DT / T_SIM, frac_cyan=res["cyan"]["k"] * DT / T_SIM,
                lyapunov_fit=fits, ghost_return_err=float(np.max(np.abs(Gh[-1, :2] - y0[:2]))),
                cyan_return_err=float(np.max(np.abs(wrap(C[-1, :2] - y0[:2])))), **chk)
    with open("sim.json", "w") as f:
        json.dump(info, f, indent=2)
    np.savez("sim.npz", F=F, C=C, O=O, ang_c=res["cyan"]["ang"], ang_o=res["orange"]["ang"],
             dist_c=res["cyan"]["dist"], dist_o=res["orange"]["dist"], env_c=res["cyan"]["env"], env_o=res["orange"]["env"])
    print(json.dumps({k: v for k, v in info.items() if k != "lyapunov_fit"}, indent=1))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
