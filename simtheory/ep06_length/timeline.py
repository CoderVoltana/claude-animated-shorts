"""Single source of truth for video time <-> sim time and every timed event (captions, sound cues, counters).
Both render.py and audio.py import this module, so pictures and sound are driven by the same clock."""
import json, math
import numpy as np

FPS = 60
DUR = 32.0
NFRAMES = 1920
RELEASE = 0.5            # video time of release (sim t = 0)
REVEAL = 17.0            # trails lift out
CHART = 23.0             # bar chart
REWIND = 29.0            # rewind starts
REWIND_END = 31.75       # pendulums back at the frame-1 pose; frames after this are identical to frame 0
TRAIL_ON = 9.5           # persistent light-painting starts (video time)
SIM_AT_REWIND = REWIND - RELEASE          # 28.5 s of sim shown before rewind
TRAIL_SIM0 = TRAIL_ON - RELEASE           # 9.0
TRAIL_SIM1 = REVEAL - RELEASE             # 16.5

DATA = "data"


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def smoother(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


def ease_out_cubic(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def sim_time(t):
    """Explicit video-time -> sim-time map. Playback is real time (1x) from release until the rewind."""
    if t < RELEASE:
        return 0.0
    if t < REWIND:
        return t - RELEASE
    if t < REWIND_END:
        return SIM_AT_REWIND * (1.0 - smoother((t - REWIND) / (REWIND_END - REWIND)))
    return 0.0


def frame_time(i):
    return i / FPS


def load():
    phys = json.load(open(f"{DATA}/physics.json"))
    sim = np.load(f"{DATA}/sim.npz")
    return phys, sim


def ranking(phys):
    """Rank by measured chaos-clock time (None = not within 60 s -> treated as +inf)."""
    items = []
    for lab, p in phys["pendulums"].items():
        tc = p["chaos_clock_s"]
        items.append((math.inf if tc is None else tc, lab))
    items.sort()
    return [lab for _, lab in items]   # most chaotic first


def fmt_chaos(tc):
    return "not within 60 s" if tc is None else f"{tc:.1f} s"


def captions(phys):
    rk = ranking(phys)
    most, least = rk[0], rk[-1]
    # (lines, t_in, t_out, slot, fade_in)  slot 'band' = strip between grid rows, 'top' = above the overlay/chart
    return [
        (["Does length", "change", "everything?"], 0.0, 2.5, "hook", False),
        (["Same angles.", "Same total length."], 2.5, 6.0, "band", True),
        (["Only the arm", "ratio changes."], 6.0, 9.5, "band", True),
        (["Same physics.", "Four personalities."], 10.0, 14.0, "band", True),
        (["Four fingerprints."], 18.0, 22.0, "top", True),
        ([f"Most chaotic: {most}."], 23.5, 26.0, "top", True),
        ([f"Most predictable: {least}."], 26.0, 29.0, "top", True),
    ]


def chart_rows(phys):
    """Bars in ranked order (most chaotic on top), each with its grow-in start time."""
    rk = ranking(phys)
    rows = []
    for j, lab in enumerate(rk):
        p = phys["pendulums"][lab]
        rows.append(dict(label=lab, tc=p["chaos_clock_s"], colour=p["colour"], t0=CHART + 0.15 + 0.45 * j))
    return rows


BAR_GROW = 0.7


def flip_events(phys):
    """Video times of every flip-counter tick shown during the forward run, per pendulum index."""
    out = []
    for k, (lab, p) in enumerate(phys["pendulums"].items()):
        for ts in p["flip_times_sim_s"]:
            if ts <= SIM_AT_REWIND:
                out.append((ts + RELEASE, k, lab))
    return sorted(out)
