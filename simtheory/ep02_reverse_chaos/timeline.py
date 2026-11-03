"""Video timeline for SimTheory #02: maps video time -> simulation time, and holds every event time.

Playback speed is the only thing that maps sim time to video time.  It is shown live in the HUD, and it is
solved here so that the divergence moments MEASURED in the simulation land on the planned video beats.
"""
import json
import os
import numpy as np
from config import FPS, DUR, T_SIM

HERE = os.path.dirname(os.path.abspath(__file__))
INFO = json.load(open(os.path.join(HERE, "sim.json")))
DT = INFO["DT"]
TAU_O = INFO["tau_orange"]          # sim seconds after reversal when ORANGE first exceeds 2 deg
TAU_C = INFO["tau_cyan"]            # same for CYAN

# ---- beats (video seconds)
T_RELEASE = 2.0
T_FREEZE = 9.0
T_FLIP = 9.9            # velocity arrows snap round
T_REWIND = 11.0
V_ORANGE = 16.4         # video time ORANGE crosses 2 deg (playback solved to land it here)
V_CYAN = 21.0           # video time CYAN crosses 2 deg
T_LAND = 28.6           # ghost arrives back at the start (at rest)
T_GRAPH = 24.0
T_GRAPH_OUT = 27.25
T_FADE_COPIES = (27.0, 28.0)
CAPTIONS = [  # (text, start, end, accent)  - the last one is the hook again for the loop
    ("Can you reverse chaos?", -1.0, 1.85, None),
    ("Let it run wild…", 2.1, 8.85, None),
    ("Now reverse every velocity.", 9.05, 10.9, "purple"),
    ("Physics says it should rewind perfectly.", 11.05, 15.85, None),
    ("Nudged by 0.0000000001°", 16.0, 19.85, "orange"),
    ("Just the computer’s rounding errors.", 20.0, 23.85, "cyan"),
    ("Tiny errors grow exponentially.", 24.0, 27.3, None),
    ("Can you reverse chaos?", 27.55, 99.0, None),
]

FINE = 2000                          # timeline resolution [samples per video second]
_tv = np.arange(int(DUR * FINE) + 1) / FINE


def _smooth(x, sec):
    k = int(sec * FINE) | 1
    w = np.hanning(k + 2)[1:-1]
    w /= w.sum()
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode="edge"), w, mode="valid")


def _sstep(a, b, t):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * u * (u * (6 * u - 15) + 10)


def _forward_speed():
    """speed profile on [T_RELEASE, T_FREEZE]: x1 at release, quick ramp to S, short ease to a freeze."""
    t = _tv
    base = (0.25 + 0.75 * _sstep(T_RELEASE, T_RELEASE + 0.7, t)) * (1 - _sstep(T_FREEZE - 0.45, T_FREEZE, t))
    base[(t < T_RELEASE) | (t > T_FREEZE)] = 0.0
    S = T_SIM / (base.sum() / FINE)
    return base * S, S


def _reverse_speed():
    """piecewise speeds S1 (aligned rewind), S2 (orange gone), S3 (cyan gone) with smooth ramps, solved so the
    measured crossings land on V_ORANGE / V_CYAN and the ghost lands at T_LAND."""
    t = _tv
    knots = [T_REWIND, V_ORANGE, V_CYAN, T_LAND]
    targets = [TAU_O, TAU_C, T_SIM]
    S = np.array([2.0, 1.5, 0.9])
    for _ in range(200):
        step = np.where(t < V_ORANGE, S[0], np.where(t < V_CYAN, S[1], S[2]))
        sp = _smooth(step, 1.2)
        sp *= _sstep(T_REWIND, T_REWIND + 0.9, t)            # start from the freeze
        sp[(t < T_REWIND) | (t > T_LAND)] = 0.0
        tau = np.cumsum(sp) / FINE
        got = [np.interp(k, t, tau) for k in knots[1:]]
        seg_have = np.diff([0.0] + got)
        seg_want = np.diff([0.0] + targets)
        if np.max(np.abs(np.array(got) - targets)) < 1e-9:
            break
        S *= (seg_want / seg_have) ** 0.8
    # exact end: tiny final rescale of the last segment only after convergence
    return sp, tau, S


FWD_SPEED, FWD_S = _forward_speed()
FWD_SIM = np.cumsum(FWD_SPEED) / FINE
FWD_SIM = np.minimum(FWD_SIM, T_SIM)
REV_SPEED, REV_TAU, REV_S = _reverse_speed()
REV_TAU = np.minimum(REV_TAU, T_SIM)
SPEED = FWD_SPEED + REV_SPEED


def phase(t):
    if t < T_RELEASE:
        return "hook"
    if t < T_FREEZE:
        return "forward"
    if t < T_REWIND:
        return "freeze"
    if t < T_LAND:
        return "rewind"
    return "landed"


def sim_forward(t):
    """forward sim time shown at video time t (also valid during the freeze)."""
    return float(np.interp(t, _tv, FWD_SIM))


def tau_rev(t):
    """sim seconds elapsed since the reversal, at video time t."""
    return float(np.interp(t, _tv, REV_TAU))


def speed(t):
    return float(np.interp(t, _tv, SPEED))


def video_time_of_tau(tau):
    return float(np.interp(tau, REV_TAU[(_tv >= T_REWIND)], _tv[_tv >= T_REWIND]))


EVENTS = dict(orange_cross=video_time_of_tau(TAU_O), cyan_cross=video_time_of_tau(TAU_C),
              land=video_time_of_tau(T_SIM - 1e-9), fwd_speed=FWD_S, rev_speeds=[float(x) for x in REV_S])

if __name__ == "__main__":
    print("forward cruise speed x%.3f" % FWD_S)
    print("reverse speeds", REV_S)
    print("events", EVENTS)
    for tt in np.arange(0, 30.01, 0.5):
        print(f"{tt:5.1f} {phase(tt):8s} speed x{speed(tt):5.2f}  fwd {sim_forward(tt):6.3f}  tau {tau_rev(tt):6.3f}")
