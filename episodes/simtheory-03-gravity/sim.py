"""SimTheory #03 - physics core.

Full nonlinear simple pendulum  theta'' = -(g/L) sin(theta), integrated with classic RK4.
No small-angle approximation anywhere in the simulation. Rigid massless rod, point mass, no damping.

Everything the video shows (angles, counters, sync time, ticks) comes from these trajectories.
Run directly to print / write the measurement log.
"""
import json
import math
import os
import numpy as np

L = 1.0                      # m
THETA0 = math.radians(20.0)  # release angle, from rest
DT = 0.0005                  # s  (0.5 ms RK4 step, spec: <= 1 ms)
G_EARTH = 9.81
G_HEAVY = 98.1               # 10x
G_GHOST = 981.0              # 100x (punchline ghost)

HERE = os.path.dirname(os.path.abspath(__file__))


def rk4(g, t_end, theta0=THETA0, dt=DT, length=L):
    """Integrate from rest at theta0. Returns t, theta, omega arrays (n+1 samples)."""
    n = int(math.ceil(t_end / dt))
    th = np.empty(n + 1)
    om = np.empty(n + 1)
    th[0], om[0] = theta0, 0.0
    k = g / length
    a, w = theta0, 0.0
    sin = math.sin
    for i in range(n):
        k1a, k1w = w, -k * sin(a)
        k2a, k2w = w + 0.5 * dt * k1w, -k * sin(a + 0.5 * dt * k1a)
        k3a, k3w = w + 0.5 * dt * k2w, -k * sin(a + 0.5 * dt * k2a)
        k4a, k4w = w + dt * k3w, -k * sin(a + dt * k3a)
        a += dt / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
        w += dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        th[i + 1], om[i + 1] = a, w
    t = np.arange(n + 1) * dt
    return t, th, om


def returns_to_release(t, th, om, g, length=L):
    """Times at which the bob is back at its release point (omega crosses 0 going + -> -, i.e. theta max).
    Each crossing is refined with Newton steps on the cubic Hermite interpolant of omega
    (omega' = -(g/L) sin theta is known exactly at the samples)."""
    k = g / length
    idx = np.where((om[1:-1] > 0) & (om[2:] <= 0))[0] + 1
    out = []
    for i in idx:
        t0, t1 = t[i], t[i + 1]
        h = t1 - t0
        w0, w1 = om[i], om[i + 1]
        d0, d1 = -k * math.sin(th[i]), -k * math.sin(th[i + 1])
        s = w0 / (w0 - w1)  # linear guess
        for _ in range(30):
            h00 = 2 * s**3 - 3 * s**2 + 1; h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2;    h11 = s**3 - s**2
            f = h00 * w0 + h10 * h * d0 + h01 * w1 + h11 * h * d1
            dh00 = 6 * s**2 - 6 * s; dh10 = 3 * s**2 - 4 * s + 1
            dh01 = -6 * s**2 + 6 * s; dh11 = 3 * s**2 - 2 * s
            fp = dh00 * w0 + dh10 * h * d0 + dh01 * w1 + dh11 * h * d1
            step = f / fp
            s -= step
            if abs(step) < 1e-15:
                break
        out.append(t0 + s * h)
    return np.array(out)


def exact_period(g, theta0=THETA0, length=L):
    """Reference only (NOT used by the simulation): T = 4 sqrt(L/g) K(sin(theta0/2)), K via AGM."""
    k = math.sin(theta0 / 2)
    a, b = 1.0, math.sqrt(1 - k * k)
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    K = math.pi / (2 * a)
    return 4 * math.sqrt(length / g) * K


class Traj:
    """Sampled trajectory with cubic-Hermite evaluation at any time (uses the exact derivative)."""

    def __init__(self, g, t_end, dt=DT):
        self.g = g
        self.dt = dt
        self.t, self.th, self.om = rk4(g, t_end, dt=dt)
        self.k = g / L
        self.al = -self.k * np.sin(self.th)
        self.t_end = self.t[-1]

    def theta(self, tt):
        tt = np.clip(np.asarray(tt, dtype=float), 0.0, self.t_end - 1e-12)
        i = np.minimum((tt / self.dt).astype(np.int64), len(self.t) - 2)
        s = tt / self.dt - i
        h = self.dt
        y0, y1 = self.th[i], self.th[i + 1]
        m0, m1 = self.om[i] * h, self.om[i + 1] * h
        h00 = 2 * s**3 - 3 * s**2 + 1; h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2;    h11 = s**3 - s**2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def measure():
    t_end = 13.0
    E = Traj(G_EARTH, t_end)
    H = Traj(G_HEAVY, t_end)
    rE = returns_to_release(E.t, E.th, E.om, G_EARTH)
    rH = returns_to_release(H.t, H.th, H.om, G_HEAVY)
    TE = rE[0]
    TH = rH[0]
    TE_mean = rE[-1] / len(rE)
    TH_mean = rH[-1] / len(rH)
    t6, t19 = rE[5], rH[18]
    sync = max(t6, t19)
    res = {
        "dt_s": DT,
        "theta0_deg": 20.0,
        "earth": {"g": G_EARTH, "period_first_s": TE, "period_mean_s": TE_mean,
                  "period_exact_elliptic_s": exact_period(G_EARTH),
                  "small_angle_period_s": 2 * math.pi * math.sqrt(L / G_EARTH),
                  "returns_s": rE.tolist()},
        "heavy": {"g": G_HEAVY, "period_first_s": TH, "period_mean_s": TH_mean,
                  "period_exact_elliptic_s": exact_period(G_HEAVY),
                  "small_angle_period_s": 2 * math.pi * math.sqrt(L / G_HEAVY),
                  "returns_s": rH.tolist()},
        "ratio": TE / TH,
        "ratio_mean": TE_mean / TH_mean,
        "sqrt10": math.sqrt(10),
        "earth_6th_return_s": t6,
        "heavy_19th_return_s": t19,
        "sync_sim_s": sync,
        "sync_mismatch_s": abs(t6 - t19),
        "sync_mismatch_pct": abs(t6 - t19) / sync * 100,
        "earth_angle_at_sync_deg": math.degrees(float(E.theta(sync))),
        "heavy_angle_at_sync_deg": math.degrees(float(H.theta(sync))),
        "ghost_period_s": exact_period(G_GHOST),
    }
    # energy drift check (no damping => energy conserved up to RK4 error)
    for name, tr in (("earth", E), ("heavy", H)):
        en = 0.5 * L**2 * tr.om**2 - tr.g * L * np.cos(tr.th)
        res[name]["energy_rel_drift"] = float(np.max(np.abs(en - en[0])) / abs(en[0]))
    return res


if __name__ == "__main__":
    r = measure()
    with open(os.path.join(HERE, "physics.json"), "w") as f:
        json.dump(r, f, indent=1)
    E, H = r["earth"], r["heavy"]
    ok_ratio = abs(r["ratio"] - 3.162) <= 0.005
    ok_sync = r["sync_mismatch_pct"] < 0.2
    print(f"RK4 dt = {DT*1000:.1f} ms, release 20.000 deg from rest, L = 1 m, no damping")
    print(f"EARTH  g= 9.81  T_sim = {E['period_first_s']:.6f} s  (mean of 6 = {E['period_mean_s']:.6f}, "
          f"elliptic ref {E['period_exact_elliptic_s']:.6f}, small-angle {E['small_angle_period_s']:.6f})")
    print(f"10x    g=98.10  T_sim = {H['period_first_s']:.6f} s  (mean of {len(H['returns_s'])} = {H['period_mean_s']:.6f}, "
          f"elliptic ref {H['period_exact_elliptic_s']:.6f}, small-angle {H['small_angle_period_s']:.6f})")
    print(f"ratio T_E/T_H = {r['ratio']:.6f}  (sqrt 10 = {r['sqrt10']:.6f})  -> 3.162 +/- 0.005: {'PASS' if ok_ratio else 'FAIL'}")
    print(f"6th Earth return  = {r['earth_6th_return_s']:.6f} s")
    print(f"19th 10x return   = {r['heavy_19th_return_s']:.6f} s")
    print(f"sync (later of the two) = {r['sync_sim_s']:.6f} s sim time, mismatch {r['sync_mismatch_s']*1000:.2f} ms = "
          f"{r['sync_mismatch_pct']:.3f}%  -> < 0.2%: {'PASS' if ok_sync else 'FAIL'}")
    print(f"angles at sync: Earth {r['earth_angle_at_sync_deg']:.4f} deg, 10x {r['heavy_angle_at_sync_deg']:.4f} deg")
    print(f"energy drift: Earth {E['energy_rel_drift']:.2e}, 10x {H['energy_rel_drift']:.2e}")
