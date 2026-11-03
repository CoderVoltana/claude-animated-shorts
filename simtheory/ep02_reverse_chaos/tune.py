"""Sweep T (and optionally start angle) and report where each reversed copy measurably separates."""
import sys, math, time
import numpy as np
from physics import *

DEG = math.pi / 180
NUDGE = 1e-10 * DEG

def run(th0, Tmax, Ts):
    n = int(round(Tmax / DT))
    Y, mi = integrate(start_state(th0, th0), DT, n)
    out = []
    for T in Ts:
        N = int(round(T / DT))
        yr = reverse(Y[N])
        C, _ = integrate(yr, DT, N)
        yo = yr.copy(); yo[0] += NUDGE
        O, _ = integrate(yo, DT, N)
        ghost = Y[N::-1]
        res = []
        for X in (O, C):
            ang = np.max(np.abs(wrap(X[:, :2] - ghost[:, :2])), axis=1) / DEG
            k = np.argmax(ang > 2.0) if np.any(ang > 2.0) else -1
            res.append(k * DT if k >= 0 else float('nan'))
        endc = np.max(np.abs(wrap(C[-1, :2] - ghost[-1, :2])))
        out.append((T, res[0], res[1], res[0] / T, res[1] / T, endc))
    return out

if __name__ == "__main__":
    th0 = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    Ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [15, 18, 20, 22, 25, 28, 30, 35]
    t0 = time.time()
    for T, to, tc, fo, fc, endc in run(th0, max(Ts), Ts):
        print(f"th0={th0:6.1f} T={T:5.1f}  orange>2deg after {to:6.3f}s ({fo:5.1%})  cyan after {tc:6.3f}s ({fc:5.1%})  cyan end err {endc:.2e} rad")
    print(f"{time.time()-t0:.1f}s")
