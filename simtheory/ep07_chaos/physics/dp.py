"""Double pendulum, RK4, generic over the number system.

The SAME code runs in float32, float64 and mpmath arbitrary precision.
Only the arithmetic type changes (the `Num` object), nothing else.
"""
import math
import numpy as np

# Physical setup (spec): equal arms 1 m, equal masses 1 kg, g = 9.81, no friction,
# released from rest at theta1 = theta2 = 120 deg, dt = 1 ms.
G_STR, L_STR, M_STR, DT_STR, ANG_DEG = "9.81", "1", "1", "0.001", 120


class Num:
    """Arithmetic backend. conv() turns a decimal string into the backend type."""
    def __init__(self, kind, dps=None):
        self.kind = kind
        self.dps = dps
        self.bits = {"f32": 24, "f64": 53}.get(kind)
        if kind == "f32":
            self.conv = lambda s: np.float32(s)
            self.sin, self.cos = np.sin, np.cos           # float32 in -> float32 out (sinf/cosf)
            self.pi = np.float32(np.pi)
            self.to_f64 = float
        elif kind == "f64":
            self.conv = lambda s: float(s)
            self.sin, self.cos = math.sin, math.cos
            self.pi = math.pi
            self.to_f64 = float
        elif kind == "mp":
            import mpmath
            self.mp = mpmath.mp.clone() if hasattr(mpmath.mp, "clone") else mpmath.mp
            # keep exactly `dps` decimal digits: mantissa of round(dps * log2(10)) bits
            # (mpmath's own dps setting silently adds about one guard digit)
            self.bits = int(round(dps * math.log2(10)))
            mpmath.mp.prec = self.bits
            self.mpmath = mpmath
            self.conv = lambda s: mpmath.mpf(s)
            self.sin, self.cos = mpmath.sin, mpmath.cos
            self.pi = mpmath.pi()
            self.to_f64 = float
        else:
            raise ValueError(kind)

    def check_type(self, x):
        if self.kind == "f32":
            assert isinstance(x, np.float32), type(x)


def make_system(num):
    c = num.conv
    g, L1, L2, m1, m2 = c(G_STR), c(L_STR), c(L_STR), c(M_STR), c(M_STR)
    two = c("2")
    half = c("0.5")
    sixth = c("1") / c("6")
    dt = c(DT_STR)
    sin, cos = num.sin, num.cos
    M = m1 + m2
    twoM1pM2 = two * m1 + m2

    def deriv(t1, t2, w1, w2):
        d = t1 - t2
        sd, cd = sin(d), cos(d)
        den = twoM1pM2 - m2 * cos(two * d)
        a1 = (-g * twoM1pM2 * sin(t1) - m2 * g * sin(t1 - two * t2)
              - two * sd * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cd)) / (L1 * den)
        a2 = (two * sd * (w1 * w1 * L1 * M + g * M * cos(t1) + w2 * w2 * L2 * m2 * cd)) / (L2 * den)
        return w1, w2, a1, a2

    def step(s):
        t1, t2, w1, w2 = s
        k1 = deriv(t1, t2, w1, w2)
        k2 = deriv(t1 + half * dt * k1[0], t2 + half * dt * k1[1], w1 + half * dt * k1[2], w2 + half * dt * k1[3])
        k3 = deriv(t1 + half * dt * k2[0], t2 + half * dt * k2[1], w1 + half * dt * k2[2], w2 + half * dt * k2[3])
        k4 = deriv(t1 + dt * k3[0], t2 + dt * k3[1], w1 + dt * k3[2], w2 + dt * k3[3])
        return tuple(s[i] + dt * sixth * (k1[i] + two * k2[i] + two * k3[i] + k4[i]) for i in range(4))

    def energy(s):
        t1, t2, w1, w2 = s
        T = half * m1 * L1 * L1 * w1 * w1 + half * m2 * (L1 * L1 * w1 * w1 + L2 * L2 * w2 * w2
                                                          + two * L1 * L2 * w1 * w2 * cos(t1 - t2))
        V = -M * g * L1 * cos(t1) - m2 * g * L2 * cos(t2)
        return T + V

    # initial state: 120 degrees converted in the backend's own precision
    th0 = c(str(ANG_DEG)) * num.pi / c("180")
    zero = c("0")
    s0 = (th0, th0, zero, zero)
    return step, energy, s0
