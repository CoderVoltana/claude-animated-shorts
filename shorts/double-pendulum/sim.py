"""Double pendulum ensemble: exact equations of motion (two point masses on
massless rigid rods, frictionless hinges, uniform gravity), integrated with
classical RK4 in float64.

Angles are measured from the downward vertical, counter-clockwise positive.
Pendulum i starts with theta2 = THETA2_0 + i * DELTA, everything else equal.
"""
import json, sys
import numpy as np

G = 9.81
L1 = L2 = 1.0
M1 = M2 = 1.0

def accel(t1, t2, w1, w2):
    d = t1 - t2
    den = 2 * M1 + M2 - M2 * np.cos(2 * d)
    a1 = (-G * (2 * M1 + M2) * np.sin(t1) - M2 * G * np.sin(t1 - 2 * t2)
          - 2 * np.sin(d) * M2 * (w2 * w2 * L2 + w1 * w1 * L1 * np.cos(d))) / (L1 * den)
    a2 = (2 * np.sin(d) * (w1 * w1 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(t1)
          + w2 * w2 * L2 * M2 * np.cos(d))) / (L2 * den)
    return a1, a2

def rk4(s, h):
    t1, t2, w1, w2 = s
    def f(t1, t2, w1, w2):
        a1, a2 = accel(t1, t2, w1, w2)
        return w1, w2, a1, a2
    k1 = f(t1, t2, w1, w2)
    k2 = f(*(x + 0.5 * h * k for x, k in zip(s, k1)))
    k3 = f(*(x + 0.5 * h * k for x, k in zip(s, k2)))
    k4 = f(*(x + h * k for x, k in zip(s, k3)))
    return tuple(x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4))

def energy(t1, t2, w1, w2):
    T = 0.5 * M1 * (L1 * w1) ** 2 + 0.5 * M2 * ((L1 * w1) ** 2 + (L2 * w2) ** 2
         + 2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2))
    V = -(M1 + M2) * G * L1 * np.cos(t1) - M2 * G * L2 * np.cos(t2)
    return T + V

def run(th1_deg, th2_deg, delta_deg, n, t_end, rate=240, sub=10):
    i = np.arange(n)
    t1 = np.full(n, np.radians(th1_deg))
    t2 = np.radians(th2_deg) + np.radians(delta_deg) * i
    s = (t1, t2, np.zeros(n), np.zeros(n))
    steps = int(round(t_end * rate))
    h = 1.0 / (rate * sub)
    out = np.empty((steps + 1, 2, n))
    out[0] = s[0], s[1]
    e0 = energy(*s)
    for k in range(1, steps + 1):
        for _ in range(sub):
            s = rk4(s, h)
        out[k] = s[0], s[1]
    drift = np.abs(energy(*s) - e0).max()
    return out, drift

def tip(th):
    x = L1 * np.sin(th[:, 0]) + L2 * np.sin(th[:, 1])
    y = L1 * np.cos(th[:, 0]) + L2 * np.cos(th[:, 1])
    return x, y

if __name__ == "__main__":
    th1, th2, delta, n, t_end = map(float, sys.argv[1:6])
    out, drift = run(th1, th2, delta, int(n), t_end)
    x, y = tip(out)
    sep = np.hypot(x - x[:, :1], y - y[:, :1]).max(axis=1)
    print("energy drift", drift)
    for t in range(0, int(t_end) + 1):
        print(t, "%.3e" % sep[t * 240])
    if len(sys.argv) > 6:
        np.save(sys.argv[6], out.astype(np.float64))
