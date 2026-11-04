"""SimTheory #04 physics core: true double-pendulum equations of motion with linear viscous joint damping.

Equal masses m = 1 kg, equal arms l = 1 m, g = 9.81 m/s^2. Absolute angles from the downward vertical.
Generalised forces from the Rayleigh dissipation function R = 1/2 c w1^2 + 1/2 c (w2 - w1)^2:
    joint 1 torque = -c*w1 (on arm 1), joint 2 torque = -c*(w2 - w1) (on arm 2, reaction on arm 1)
    Q1 = -c*w1 + c*(w2 - w1),  Q2 = -c*(w2 - w1),  dE/dt = -2R <= 0.
Integrator: classical RK4, float64, fixed step.
"""
import numpy as np

G = 9.81
L = 1.0
M = 1.0


def joint_torque(w, c, mu, w0):
    """Friction torque at a joint with relative angular velocity w: viscous -c*w plus regularised dry (Coulomb)
    friction -mu*tanh(w/w0). Always opposes w, so it can only remove energy."""
    return -c * w - mu * np.tanh(w / w0)


def deriv(s, c, mu=0.0, w0=1.0):
    """s: (..., 4) = th1, th2, w1, w2. c, mu, w0: friction params (broadcastable)."""
    th1, th2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = th1 - th2
    cd, sd = np.cos(d), np.sin(d)
    ml2 = M * L * L
    tau1 = joint_torque(w1, c, mu, w0)        # joint 1 (pivot): acts on arm 1
    tau2 = joint_torque(w2 - w1, c, mu, w0)   # joint 2 (elbow): +tau2 on arm 2, -tau2 on arm 1
    q1 = tau1 - tau2
    q2 = tau2
    # mass matrix [[2, cd], [cd, 1]] * ml2
    r1 = -ml2 * w2 * w2 * sd - 2 * M * G * L * np.sin(th1) + q1
    r2 = ml2 * w1 * w1 * sd - M * G * L * np.sin(th2) + q2
    det = ml2 * (2.0 - cd * cd)
    a1 = (r1 - cd * r2) / det
    a2 = (2.0 * r2 - cd * r1) / det
    return np.stack([w1, w2, a1, a2], axis=-1)


def rk4_step(s, c, dt, mu=0.0, w0=1.0):
    k1 = deriv(s, c, mu, w0)
    k2 = deriv(s + 0.5 * dt * k1, c, mu, w0)
    k3 = deriv(s + 0.5 * dt * k2, c, mu, w0)
    k4 = deriv(s + dt * k3, c, mu, w0)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def energy(s):
    th1, th2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    T = M * L * L * (w1 * w1 + 0.5 * w2 * w2 + w1 * w2 * np.cos(th1 - th2))
    V = -2 * M * G * L * np.cos(th1) - M * G * L * np.cos(th2)
    return T + V


E_REST = -3 * M * G * L  # minimum energy: hanging straight down


def wrap180(a_deg):
    return (a_deg + 180.0) % 360.0 - 180.0


def bob_positions(s):
    th1, th2 = s[..., 0], s[..., 1]
    x1, y1 = L * np.sin(th1), L * np.cos(th1)  # y downward
    x2, y2 = x1 + L * np.sin(th2), y1 + L * np.cos(th2)
    return x1, y1, x2, y2


def simulate(c_list, offset_deg, T, dt, record_every, theta0_deg=120.0):
    """Simulate twin pairs. Returns states (n_rec, N, 4) with N = 2*len(c_list): [p0a, p0b, p1a, p1b, ...]."""
    c_arr = np.repeat(np.asarray(c_list, dtype=np.float64), 2)
    N = len(c_arr)
    s = np.zeros((N, 4))
    s[:, 0] = np.radians(theta0_deg)
    s[:, 1] = np.radians(theta0_deg)
    s[1::2, 0] += np.radians(offset_deg)
    nsteps = int(round(T / dt))
    rec = [s.copy()]
    Eprev = energy(s)
    max_rel_rise = np.zeros(N)
    scale = energy(s) - E_REST
    for i in range(1, nsteps + 1):
        s = rk4_step(s, c_arr, dt)
        E = energy(s)
        max_rel_rise = np.maximum(max_rel_rise, (E - Eprev) / scale)
        Eprev = E
        if i % record_every == 0:
            rec.append(s.copy())
    return np.array(rec), max_rel_rise
