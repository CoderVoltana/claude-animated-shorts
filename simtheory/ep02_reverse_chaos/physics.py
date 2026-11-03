"""SimTheory #02 - Can you reverse chaos?   Physics core.

Ideal double pendulum: two equal rigid massless rods (L = 1 m), two equal point masses (m = 1 kg),
g = 9.81 m/s^2, no damping, no friction.  float64 throughout.

Integrator: IMPLICIT MIDPOINT on Hamilton's equations in canonical coordinates y = (th1, th2, p1, p2)
    y_{n+1} = y_n + h * f((y_n + y_{n+1}) / 2)
solved by fixed-point iteration until the iterate stops changing (machine precision).  The method is
time-symmetric, R-reversible for R:(th, p) -> (th, -p) and symplectic.  Because p = M(th) * omega with
M depending only on the angles, p -> -p is EXACTLY omega -> -omega (negation is exact in IEEE float).
"""
import math
import numpy as np
from numba import njit

G = 9.81
L = 1.0
M = 1.0
DT = 1.0e-4            # integrator step [s]


@njit(cache=True)
def rhs(t1, t2, p1, p2):
    """Hamilton's equations for the equal-mass, equal-length double pendulum."""
    d = t1 - t2
    c = math.cos(d)
    s = math.sin(d)
    den = M * L * L * (1.0 + s * s)
    w1 = (p1 - p2 * c) / den
    w2 = (2.0 * p2 - p1 * c) / den
    C1 = p1 * p2 * s / den
    C2 = (p1 * p1 + 2.0 * p2 * p2 - 2.0 * p1 * p2 * c) * math.sin(2.0 * d) / (2.0 * M * L * L * (1.0 + s * s) ** 2)
    dp1 = -2.0 * M * G * L * math.sin(t1) - C1 + C2
    dp2 = -M * G * L * math.sin(t2) + C1 - C2
    return w1, w2, dp1, dp2


@njit(cache=True)
def omegas(t1, t2, p1, p2):
    d = t1 - t2
    c = math.cos(d)
    s = math.sin(d)
    den = M * L * L * (1.0 + s * s)
    return (p1 - p2 * c) / den, (2.0 * p2 - p1 * c) / den


@njit(cache=True)
def momenta(t1, t2, w1, w2):
    c = math.cos(t1 - t2)
    return M * L * L * (2.0 * w1 + w2 * c), M * L * L * (w2 + w1 * c)


@njit(cache=True)
def energy(t1, t2, p1, p2):
    d = t1 - t2
    c = math.cos(d)
    s = math.sin(d)
    T = (p1 * p1 + 2.0 * p2 * p2 - 2.0 * p1 * p2 * c) / (2.0 * M * L * L * (1.0 + s * s))
    V = -M * G * L * (2.0 * math.cos(t1) + math.cos(t2))
    return T + V


@njit(cache=True)
def step(y, h):
    """One implicit-midpoint step, fixed-point iterated to machine precision. Returns (y_new, iterations)."""
    a0, a1, a2, a3 = y[0], y[1], y[2], y[3]
    f0, f1, f2, f3 = rhs(a0, a1, a2, a3)
    b0, b1, b2, b3 = a0 + h * f0, a1 + h * f1, a2 + h * f2, a3 + h * f3     # explicit Euler guess
    best = 1e300
    stall = 0
    it = 0
    for it in range(1, 100):
        f0, f1, f2, f3 = rhs(0.5 * (a0 + b0), 0.5 * (a1 + b1), 0.5 * (a2 + b2), 0.5 * (a3 + b3))
        n0, n1, n2, n3 = a0 + h * f0, a1 + h * f1, a2 + h * f2, a3 + h * f3
        diff = max(abs(n0 - b0), abs(n1 - b1), abs(n2 - b2), abs(n3 - b3))
        b0, b1, b2, b3 = n0, n1, n2, n3
        if diff == 0.0:
            break
        if diff < best:
            best = diff
            stall = 0
        else:                       # iterate is bouncing at the last bit: converged to rounding
            stall += 1
            if stall >= 3:
                break
    out = np.empty(4)
    out[0] = b0
    out[1] = b1
    out[2] = b2
    out[3] = b3
    return out, it


@njit(cache=True)
def integrate(y0, h, n):
    """Integrate n steps; returns every state (n+1, 4) and the max fixed-point iteration count."""
    Y = np.empty((n + 1, 4))
    Y[0] = y0
    y = y0.copy()
    maxit = 0
    for i in range(n):
        y, it = step(y, h)
        if it > maxit:
            maxit = it
        Y[i + 1] = y
    return Y, maxit


@njit(cache=True)
def energies(Y):
    E = np.empty(Y.shape[0])
    for i in range(Y.shape[0]):
        E[i] = energy(Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3])
    return E


@njit(cache=True)
def all_omegas(Y):
    W = np.empty((Y.shape[0], 2))
    for i in range(Y.shape[0]):
        w1, w2 = omegas(Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3])
        W[i, 0] = w1
        W[i, 1] = w2
    return W


def reverse(y):
    """omega -> -omega, exactly (p is linear in omega with angle-only coefficients)."""
    return np.array([y[0], y[1], -y[2], -y[3]])


def start_state(th1_deg, th2_deg):
    return np.array([math.radians(th1_deg), math.radians(th2_deg), 0.0, 0.0])


def bobs(th1, th2):
    """Bob positions, pivot at origin, y up."""
    x1 = L * np.sin(th1)
    y1 = -L * np.cos(th1)
    return x1, y1, x1 + L * np.sin(th2), y1 - L * np.cos(th2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lagrange_accel(t1, t2, w1, w2):
    """Textbook omega-form accelerations, used only to cross-check the Hamiltonian form."""
    m1 = m2 = M
    l1 = l2 = L
    d = t1 - t2
    den = 2 * m1 + m2 - m2 * math.cos(2 * t1 - 2 * t2)
    a1 = (-G * (2 * m1 + m2) * math.sin(t1) - m2 * G * math.sin(t1 - 2 * t2)
          - 2 * math.sin(d) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(d))) / (l1 * den)
    a2 = (2 * math.sin(d) * (w1 * w1 * l1 * (m1 + m2) + G * (m1 + m2) * math.cos(t1)
          + w2 * w2 * l2 * m2 * math.cos(d))) / (l2 * den)
    return a1, a2
