"""SimTheory #03 - one timeline shared by the picture (render.py) and the sound (audio.py).

All physics-driven times (ticks, counter pops, the sync freeze) come from sim.py, nothing is hand-placed.
"""
import math
import numpy as np
import sim

FPS = 60
DUR = 30.0                      # s, 1800 frames
NFRAMES = int(round(DUR * FPS))
T_REL = 2.00                    # release, exactly 2.00 s

_phys = sim.measure()
RET_E = np.array(_phys["earth"]["returns_s"])       # sim-time returns to the release point
RET_H = np.array(_phys["heavy"]["returns_s"])
SYNC_SIM = _phys["sync_sim_s"]                      # = 19th heavy return (later of 6th Earth / 19th heavy)
T_SYNC = T_REL + SYNC_SIM                           # video time of the freeze (~14.146 s)
PHYS = _phys

# counter events in video time (only those up to the sync point)
CNT_E = T_REL + RET_E[RET_E <= SYNC_SIM + 1e-9]     # 6 events
CNT_H = T_REL + RET_H[RET_H <= SYNC_SIM + 1e-9]     # 19 events
assert len(CNT_E) == 6 and len(CNT_H) == 19

# ---- reveal / punchline / loop -------------------------------------------------------------
T_FORMULA = T_SYNC + 1.40       # formula starts animating in, after the chord + brief silence
FORMULA_GROUPS = [("T", 0.00), ("eq", 0.12), ("twopi", 0.24), ("root", 0.42),
                  ("L", 0.62), ("slash", 0.70), ("g", 0.78)]
T_ROOT_GLOW = T_FORMULA + 0.95  # radical starts glowing / pulsing
T_FORMULA_OUT = (21.55, 21.95)

CAM_IN = (T_SYNC + 0.9, 20.7)   # slow push-in on the formula
CAM_OUT = (20.8, 23.0)          # ease back out before the loop
CAM_ZOOM = 1.09

GHOST_REL = 21.45               # ghost (100x g) sim starts here (already swinging when it appears)
GHOST_IN = (21.85, 22.35)       # flicker in
GHOST_OUT = (26.45, 27.15)      # flicker / fade out
GHOST_ALPHA = 0.62

T_RESET = 27.45                 # counters reset to 0

# captions: (text, t_in, t_out); fade in/out are short and smooth
CAPTIONS = [
    ("10× stronger gravity…", -10.0, 1.86),
    ("…10× faster swing?", 2.00, 3.92),
    ("Count the swings.", 4.05, T_SYNC - 0.02),
    ("Only 3.16× faster.", T_SYNC + 0.18, 17.95),
    ("Gravity hides under a square root.", 18.10, 21.70),
    ("For 10× faster, you'd need 100× gravity.", 21.95, 27.15),
    ("10× stronger gravity…", 27.70, 99.0),
]
CAP_FADE_IN = 0.22
CAP_FADE_OUT = 0.18


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


def cam_zoom(t):
    if t < CAM_OUT[0]:
        return 1.0 + (CAM_ZOOM - 1.0) * smoother((t - CAM_IN[0]) / (CAM_IN[1] - CAM_IN[0]))
    return CAM_ZOOM - (CAM_ZOOM - 1.0) * smoother((t - CAM_OUT[0]) / (CAM_OUT[1] - CAM_OUT[0]))


_FLICK_IN = [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
_FLICK_OUT = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def ghost_alpha(t):
    """0..1 visibility of the 100x ghost (flicker in, steady translucent, flicker/fade out)."""
    if t < GHOST_IN[0] or t >= GHOST_OUT[1]:
        return 0.0
    if t < GHOST_IN[1]:
        u = (t - GHOST_IN[0]) / (GHOST_IN[1] - GHOST_IN[0])
        k = min(int(u * len(_FLICK_IN)), len(_FLICK_IN) - 1)
        return _FLICK_IN[k] * (0.55 + 0.45 * u)
    if t < GHOST_OUT[0]:
        return 1.0
    u = (t - GHOST_OUT[0]) / (GHOST_OUT[1] - GHOST_OUT[0])
    k = min(int(u * len(_FLICK_OUT)), len(_FLICK_OUT) - 1)
    return _FLICK_OUT[k] * (1.0 - u) ** 1.5
