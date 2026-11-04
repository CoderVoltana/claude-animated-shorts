"""Step 2: the single source of truth for TIME.

Explicit mapping between video time (tv, seconds, frame/60) and the simulation clocks:
  * live clock    : sim = tv - 0.5                      (0.5 s .. 40 s, real time 1x)
  * replay clock  : sim = (tv - 28) * 0.5               (28 s .. 38 s, 0.5x slow motion, sim 0 -> 5 s)
  * rewind clock  : sim = 39.5 * (1 - ease(u))          (40 s .. 41.7 s, fast reverse to the push pose)
Every caption, sound cue and readout is driven from the functions in this file.
"""
import math

import numpy as np

from config import FRACS, FPS, NFRAMES, PIVOT, THR, TOP, W0

T_PUSH = 0.5
REPLAY_A, REPLAY_B, REPLAY_RATE = 28.0, 38.0, 0.5
REW_A, REW_B = 40.0, 41.7
LIVE_END = REW_A - T_PUSH          # 39.5 s of sim time at the start of the rewind


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def lerp(a, b, s):
    return a + (b - a) * s


def window(tv, a, b, fin=0.2, fout=0.2):
    """1 inside [a, b] with linear-ish smooth fades of fin / fout seconds (fade-in starts at a, fade-out ends at b)."""
    if tv < a or tv > b:
        return 0.0
    return smooth((tv - a) / fin if fin > 0 else 1.0) * smooth((b - tv) / fout if fout > 0 else 1.0)


def frame_time(fi):
    return fi / FPS


# ------------------------------------------------------------------ clocks
def rewind_sim(tv):
    u = clamp((tv - REW_A) / (REW_B - REW_A))
    return LIVE_END * (1.0 - (0.5 - 0.5 * math.cos(math.pi * u)))


def is_rest(tv):
    """Hook pose: everything hanging straight down at rest (frame 0 .. push, and the end of the loop)."""
    return tv < T_PUSH or tv >= REW_B


def layers(tv):
    """Pendulum-view layers to draw: list of dicts(t=sim time, w=weight, rate=sim s per video s, mode, thr=bool, thr_w).
    Two layers only during short cross-fades between clocks."""
    if is_rest(tv):
        return [dict(t=0.0, w=1.0, rate=0.0, mode="rest", thr=True, thr_w=1.0)]
    if tv < 27.5:
        return [dict(t=tv - T_PUSH, w=1.0, rate=1.0, mode="live", thr=True, thr_w=1.0)]
    if tv < 28.1:                                      # live -> replay crossfade during the zoom back
        s = smooth((tv - 27.5) / 0.6)
        return [dict(t=tv - T_PUSH, w=1 - s, rate=1.0, mode="live", thr=True, thr_w=1 - s),
                dict(t=max(0.0, (tv - REPLAY_A) * REPLAY_RATE), w=s, rate=REPLAY_RATE if tv >= REPLAY_A else 0.0,
                     mode="replay", thr=True, thr_w=s)]
    if tv < 37.6:
        return [dict(t=(tv - REPLAY_A) * REPLAY_RATE, w=1.0, rate=REPLAY_RATE, mode="replay", thr=True, thr_w=1.0)]
    if tv < 38.4:                                      # dimmed others cross-fade replay -> live; threshold switches at 38.0
        s = smooth((tv - 37.6) / 0.8)
        return [dict(t=(tv - REPLAY_A) * REPLAY_RATE, w=1 - s, rate=REPLAY_RATE, mode="replay", thr=tv < REPLAY_B, thr_w=1.0),
                dict(t=tv - T_PUSH, w=s, rate=1.0, mode="live", thr=tv >= REPLAY_B, thr_w=1.0)]
    if tv < REW_A:
        return [dict(t=tv - T_PUSH, w=1.0, rate=1.0, mode="live", thr=True, thr_w=1.0)]
    # rewind
    t = rewind_sim(tv)
    t_prev = rewind_sim(tv - 1.0 / FPS)
    return [dict(t=t, w=1.0, rate=-(t_prev - t) * FPS, mode="rewind", thr=True, thr_w=1.0,
                 t_prev=t_prev, u=clamp((tv - REW_A) / (REW_B - REW_A)))]


def readout_clock(tv):
    """(label, clock_seconds) for the gap readout, or None when the readout is hidden."""
    if REPLAY_A <= tv < REPLAY_B:
        return "replay", (tv - REPLAY_A) * REPLAY_RATE
    if REPLAY_B <= tv < REW_A + 0.35:
        return "live", min(tv, REW_A) - T_PUSH
    return None


def readout_alpha(tv):
    return window(tv, 27.9, REW_A + 0.35, 0.3, 0.35)


def replay_label_alpha(tv):
    return window(tv, 27.9, REPLAY_B, 0.3, 0.2)


def live_label_alpha(tv):
    return window(tv, REPLAY_B, REW_A + 0.35, 0.2, 0.35)


# ------------------------------------------------------------------ gap readout (computed DIRECTLY from delta)
def gap_deg(t):
    return math.degrees(4.0 * math.atan(math.exp(-W0 * t)))


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def gap_parts(t):
    """Readout pieces: (mantissa_text, exponent_text or None). Decimals while >= 0.001 deg, then scientific notation."""
    d = gap_deg(t)
    if d >= 10:
        return f"{d:.1f}", None
    if d >= 1:
        return f"{d:.2f}", None
    if d >= 1e-3:
        e = math.floor(math.log10(d))
        dec = -e + 1                  # two significant figures
        return f"{d:.{dec}f}", None
    e = math.floor(math.log10(d))
    m = d / 10 ** e
    if round(m, 1) >= 10:
        m /= 10
        e += 1
    return f"{m:.1f} × 10", f"{e:d}".replace("-", "−")


def gap_string(t):
    m, e = gap_parts(t)
    if e is None:
        return f"Gap to top: {m}°"
    return f"Gap to top: {m}{e.replace('−', '-').translate(SUP)} °"


# ------------------------------------------------------------------ per-pendulum emphasis
def pend_alpha(i, tv):
    trapped = FRACS[i] < 1.0
    a = 1.0
    # 8.0-12.5: focus on the trapped swingers
    s1 = smooth((tv - 7.8) / 0.5) * (1 - smooth((tv - 12.3) / 0.4))
    if not trapped:
        a = lerp(a, 0.32, s1)
    # 12.5-17.5: focus on the flippers
    s2 = smooth((tv - 12.3) / 0.4) * (1 - smooth((tv - 16.9) / 0.5))
    if trapped or i == THR:
        a = lerp(a, 0.32, s2)
    # replay: everything but the threshold pendulum dims to 20 %
    s3 = smooth((tv - 27.9) / 0.7) * (1 - smooth((tv - REW_A) / 1.2))
    if i != THR:
        a = lerp(a, 0.2, s3)
    return a


def trapped_ticks_alpha(tv):
    return window(tv, 9.6, 12.5, 0.4, 0.3)


# ------------------------------------------------------------------ camera
INSET_C = np.array([1030.0, 1520.0])
INSET_K = 115.0 / 570.0
ZOOM_C = np.array([619.0, 1120.0])     # screen position of the top marker during the replay
ZOOM_K0, ZOOM_K = 1.2, 2.3


def camera(tv):
    """Pendulum-view camera: screen = C + k * (world - F). Returns (k, F, C)."""
    P = PIVOT.astype(float)
    T = TOP.astype(float)
    if tv < 17.5 or tv >= REW_B:
        return 1.0, P, P
    if tv < 27.5:                                     # pull out into the corner inset (map reveal)
        s = smoother((tv - 17.5) / 0.6)
        return lerp(1.0, INSET_K, s), P, lerp(P, INSET_C, s)
    if tv < 28.6:                                     # inset grows back into the replay framing (top marker fixed)
        s = smoother((tv - 27.5) / 1.1)
        return lerp(INSET_K, ZOOM_K0, s), lerp(P, T, s), lerp(INSET_C, ZOOM_C, s)
    # slow push-in on the top marker during the replay (marker stays put on screen)
    s = smooth((tv - 28.6) / (REPLAY_B - 28.6))
    k = lerp(ZOOM_K0, ZOOM_K, s) + 0.06 * clamp((tv - REPLAY_B) / 2.0)
    F, C = T, ZOOM_C
    if tv >= REW_A:                                   # ease out back to the hook framing
        e = smoother((tv - REW_A) / (REW_B - REW_A - 0.05))
        k = lerp(k, 1.0, e)
        F, C = lerp(F, P, e), lerp(C, P, e)
    return k, F, C


# ------------------------------------------------------------------ phase portrait
FULL_RECT = (124.0, 790.0, 990.0, 1641.0)      # x0, y(omega=+10), x1, y(omega=-8.5)
MINI_RECT = (126.0, 1440.0, 520.0, 1636.0)
OM_TOP, OM_BOT = 10.0, -8.5
CURVE_A = 17.5                                    # sim time when the map starts drawing (video 18.0 s)


def plot_rect(tv):
    if tv < 27.5:
        s = smooth((tv - 17.6) / 0.6)
        sc = lerp(1.08, 1.0, s)                       # gentle pull-out as the map appears
        x0, y0, x1, y1 = FULL_RECT
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (cx + (x0 - cx) * sc, cy + (y0 - cy) * sc, cx + (x1 - cx) * sc, cy + (y1 - cy) * sc), 0.0
    s = smoother((tv - 27.5) / 0.7)
    return tuple(lerp(a, b, s) for a, b in zip(FULL_RECT, MINI_RECT)), s


def plot_alpha(tv):
    return window(tv, 17.6, REW_A + 0.3, 0.6, 0.5)


def separatrix_progress(tv):
    return clamp((tv - 23.0) / 2.0)


def separatrix_glow(tv):
    """0..1 'brightest thing on screen' ramp after the trace completes, with a soft pulse."""
    g = smooth((tv - 24.9) / 0.6)
    pulse = 0.25 * math.exp(-((tv - 25.4) / 0.35) ** 2)
    return g + pulse


def eye_labels_alpha(tv):
    return window(tv, 25.6, 27.5, 0.4, 0.3)


def curves_dim(tv):
    return lerp(1.0, 0.55, smooth((tv - 24.8) / 0.6))


# ------------------------------------------------------------------ captions / text
CAPTIONS = [
    ("Same pendulum.\n9 different pushes.", 3.0, 7.5),
    ("Too little energy:\ntrapped.", 8.0, 12.5),
    ("Enough energy:\nit flips forever.", 12.5, 17.5),
    ("Every motion\nhas a map.", 18.5, 22.5),
    ("Inside the eye: trapped.\nOutside: flipping.", 23.0, 27.5),
    ("Exactly enough\nenergy…", 29.0, 33.0),
    ("…it slows down\nforever.", 33.5, 37.5),
    ("Exactly enough.\nNever reaches the top.", 38.0, 40.0),
]


def caption_alphas(tv):
    out = []
    for txt, a, b in CAPTIONS:
        al = window(tv, a, b, 0.2, 0.2)
        if al > 0:
            out.append((txt, al, a))
    return out


def hook_alpha(tv):
    if tv < 3.0:
        return 1.0 - smooth((tv - 2.8) / 0.2)
    return smooth((tv - 41.25) / 0.4)


def legend_alpha(tv):
    return window(tv, 3.3, 17.4, 0.5, 0.4)


def flash_window(tv):
    return window(tv, 12.5, 17.4, 0.3, 0.3)


# ------------------------------------------------------------------ checks used by QA
def word_count(s):
    return len(s.replace("\n", " ").replace("…", " ").split())


if __name__ == "__main__":
    for t in (1, 3, 5, 37.5, 39.5):
        print(t, gap_deg(t), gap_string(t))
    for txt, a, b in CAPTIONS:
        print(repr(txt), word_count(txt), b - a)
    for fi in (0, NFRAMES - 1):
        print(fi, layers(frame_time(fi)), camera(frame_time(fi)))
