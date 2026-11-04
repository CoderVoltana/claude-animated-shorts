"""SimTheory #04 master clock: explicit video-time -> sim-time mapping shared by renderer, HUD, captions and audio.

Video: 34.0 s, 60 fps, 2040 frames. Release at video 0.5 s.
Playback speed (shown in the HUD at all times):
  0.5 -> 10.5 s   1x real time (the whole chaotic phase)
  10.5 -> 14.5 s  smooth ramp up to RMAX x  (time-lapse, labelled)
  14.5 -> 23.5 s  RMAX x
  23.5 -> 25.5 s  smooth ramp back down to 1x
  25.5 -> 31.0 s  1x
  31.0 -> 33.85 s fast reverse rewind of the recorded motion back to sim time 0
  33.85 -> 34.0 s hold on the frame-1 pose (last frame == first frame)
RMAX is solved so that the measured friction-twin re-merge lands at video MERGE_TARGET.
"""
import json, os
import numpy as np

FPS = 60
DUR = 34.0
NFR = int(round(DUR * FPS))  # 2040
T_REL = 0.5
T_UP0, T_UP1 = 10.5, 14.5
T_DN0, T_DN1 = 23.5, 25.5
T_RW0, T_RW1 = 31.0, 33.85
MERGE_TARGET = 21.0

HERE = os.path.dirname(os.path.abspath(__file__))
_cfg_path = os.path.join(HERE, "data", "timemap.json")
RMAX = json.load(open(_cfg_path))["rmax"] if os.path.exists(_cfg_path) else 7.5


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smooth_int(u):
    """integral_0^u smooth(v) dv"""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 - 0.5 * u ** 4


def rate(t, rmax=None):
    """Forward playback rate (sim seconds per video second) for t < T_RW0."""
    R = RMAX if rmax is None else rmax
    t = np.asarray(t, dtype=np.float64)
    r = np.where(t < T_REL, 0.0, 1.0)
    up = smooth((t - T_UP0) / (T_UP1 - T_UP0))
    dn = smooth((t - T_DN0) / (T_DN1 - T_DN0))
    r = np.where(t >= T_REL, 1.0 + (R - 1.0) * (up - dn), r)
    return r


def sim_forward(t, rmax=None):
    """Closed-form integral of rate(t) from 0 to t (valid for t <= T_RW0)."""
    R = RMAX if rmax is None else rmax
    t = np.asarray(t, dtype=np.float64)
    s = np.maximum(t - T_REL, 0.0)
    wu, wd = T_UP1 - T_UP0, T_DN1 - T_DN0
    s = s + (R - 1.0) * (wu * smooth_int((t - T_UP0) / wu) + np.maximum(t - T_UP1, 0.0))
    s = s - (R - 1.0) * (wd * smooth_int((t - T_DN0) / wd) + np.maximum(t - T_DN1, 0.0))
    return s


def rewind_u(t):
    return np.clip((np.asarray(t, dtype=np.float64) - T_RW0) / (T_RW1 - T_RW0), 0.0, 1.0)


def rewind_ease(u):
    # ease-in-out (quintic smootherstep): starts gently, rushes, lands softly on the frame-1 pose
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def sim_time(t, rmax=None):
    t = np.asarray(t, dtype=np.float64)
    s_end = sim_forward(T_RW0, rmax)
    fwd = sim_forward(np.minimum(t, T_RW0), rmax)
    back = s_end * (1.0 - rewind_ease(rewind_u(t)))
    return np.where(t < T_RW0, fwd, back)


def sim_end(rmax=None):
    return float(sim_forward(T_RW0, rmax))


def frame_times():
    return np.arange(NFR) / FPS
