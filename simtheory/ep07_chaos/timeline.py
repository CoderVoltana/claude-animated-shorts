"""Single source of truth for timing: video time <-> sim time, captions, events, sound cues.

Everything that is shown or heard is driven from here. Numbers on screen come from
data/measurements.json (written by physics/analyze.py), never typed by hand.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------------------------------------------------------------- format
W, H = 1440, 2560
FPS = 60
NF = 36 * FPS                     # 2160 frames
T_LAST = (NF - 1) / FPS           # 35.98333 s, time of the last frame
SAFE = (86, 384, 1152, 1664)      # x0, y0, x1, y1: all text must sit inside
SAFE_CX = (SAFE[0] + SAFE[2]) / 2  # 619
SAFE_CY = (SAFE[1] + SAFE[3]) / 2  # 1024

# ---------------------------------------------------------------- measurements
M = json.load(open(os.path.join(DATA, "measurements.json")))
HOR = {k: v["horizon_s"] for k, v in M["horizons"].items()}
H7, H16 = HOR["f32"], HOR["f64"]
SLOPE = M["fit"]["slope_s_per_digit"]
INTERCEPT = M["fit"]["intercept_s"]
D60 = M["fit"]["digits_for_60s"]
CHART_POINTS = [(M["horizons"][k]["digits"], HOR[k], k) for k in ["f32", "f64", "mp24", "mp32", "mp40"]]

# ---------------------------------------------------------------- playback mapping
RELEASE = 0.5                     # pendulums released at 0.5 s video time
SPEED = 1.3                       # constant fast-forward of the pendulum section (HUD "x1.3")
REWIND_T0 = 33.0                  # rewind starts
REWIND_T1 = 35.70                 # rewind arrives at the start pose; held (pulsing) until the loop
TAU_REWIND0 = (REWIND_T0 - RELEASE) * SPEED

T7 = RELEASE + H7 / SPEED         # video time the 7-digit copy crosses 1 deg (measured)
T16 = RELEASE + H16 / SPEED       # video time the 16-digit copy crosses 1 deg (measured)


def _hermite(p0, m0, p1, m1, s):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1


def loop_time(t):
    """Frames are rendered from this; the last frame maps exactly onto frame 0."""
    return 0.0 if t >= T_LAST - 1e-9 else t


def hold_phase(t):
    """Time inside the deliberate start-pose hold (continuous across the loop seam).
    0 .. 0.5 at the start; -(T_LAST-REWIND_T1) .. 0 at the end. None outside the hold."""
    t = loop_time(t)
    if t <= RELEASE:
        return t
    if t >= REWIND_T1:
        return t - T_LAST
    return None


def sim_time(t):
    """Sim time (s) shown at video time t."""
    t = loop_time(t)
    if t <= RELEASE:
        return 0.0
    if t <= REWIND_T0:
        return (t - RELEASE) * SPEED
    if t < REWIND_T1:
        d = REWIND_T1 - REWIND_T0
        s = (t - REWIND_T0) / d
        # starts at forward speed (tape-stop feel), arrives at rest at tau = 0
        return max(0.0, _hermite(TAU_REWIND0, SPEED * d * 0.35, 0.0, 0.0, s))
    return 0.0


def rewinding(t):
    t = loop_time(t)
    return REWIND_T0 <= t < REWIND_T1


# ---------------------------------------------------------------- easing helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(e0, e1, x):
    """smootherstep from e0 to e1"""
    u = clamp((x - e0) / (e1 - e0))
    return u * u * u * (u * (u * 6 - 15) + 10)


def fade_window(t, a, b, f=0.2):
    """1 inside [a, b], quick smooth fades of f seconds (fade in starts at a, fade out ends at b)."""
    return smooth(a, a + f, t) * (1 - smooth(b - f, b, t))


# ---------------------------------------------------------------- captions
def fmt1(x):
    return f"{x:.1f}"


HOOK_TEXT = "Can a computer predict chaos?"
HOOK_LINES = ["Can a computer", "predict chaos?"]

CAPTIONS = [
    (2.50, 4.90, ["Same physics.", "Same start.", "Same code."]),
    (4.90, T7 - 0.08, ["Only difference: digits", "the computer keeps."]),
    (T7 + 0.05, 13.00, ["7 digits: wrong after", f"{fmt1(H7)} seconds."]),
    (T16 + 0.05, 21.00, ["16 digits: wrong after", f"{fmt1(H16)} seconds."]),
    (21.50, 25.50, ["Every extra digit buys", "the same small slice."]),
    (25.50, 29.00, ["To predict 1 minute", f"ahead: ~{round(D60)} digits."]),
    (29.00, 33.00, ["Real measurements", "are far less precise."]),
]


def hook_alpha(t):
    t = loop_time(t)
    if t <= 2.5:
        return 1 - smooth(2.30, 2.50, t)
    return smooth(34.90, 35.50, t)


def legend_alpha(t):
    t = loop_time(t)
    return smooth(0.6, 1.4, t) * (1 - smooth(21.0, 21.6, t))


def readout_alpha(t, which):
    t = loop_time(t)
    start = {"f32": 5.4, "f64": 13.0}[which]
    return smooth(start, start + 0.4, t) * legend_alpha(t)


def orange_focus(t):
    """Orange copy steps back (dims) while we watch the cyan copy hold on."""
    t = loop_time(t)
    return 1 - 0.45 * smooth(15.2, 15.9, t) * (1 - smooth(T16 + 0.3, T16 + 1.2, t))


# ---------------------------------------------------------------- camera (zoom about the pivot)
def zoom(t):
    t = loop_time(t)
    h = hold_phase(t)
    if h is not None:
        return 1.0 + 0.010 * h            # slow push during the start-pose hold (continuous across the loop)
    if t < T7:
        # slow push-in while the three copies are together
        return 1.005 + 0.035 * (t - RELEASE) / (T7 - RELEASE)
    if t < T7 + 1.3:
        z_split = 1.04
        return z_split + (1.0 - z_split) * smooth(T7, T7 + 1.3, t)
    if t < T16:
        return 1.0 + 0.03 * smooth(T7 + 1.3, T16, t)
    if t < 21.0:
        return 1.03 + (0.985 - 1.03) * smooth(T16, T16 + 1.3, t)
    if t < REWIND_T0:
        return 0.985
    # rewind: come back to the hold zoom value at REWIND_T1
    z_end = 1.0 + 0.010 * (REWIND_T1 - T_LAST)
    return 0.985 + (z_end - 0.985) * smooth(REWIND_T0 + 0.2, REWIND_T1, t)


# chart transition: 0 = full pendulum view, 1 = chart with corner inset
def chart_mix(t):
    t = loop_time(t)
    return smooth(21.0, 22.4, t) * (1 - smooth(REWIND_T0, REWIND_T0 + 1.2, t))


# ---------------------------------------------------------------- chart choreography
CHART_POINT_TIMES = [22.2, 23.0, 23.8, 24.6, 25.4]
FIT_T0, FIT_T1 = 25.6, 26.8
SLOPE_LABEL_T = 26.5
MINUTE_T = 27.3          # gold 60-second line + ~N digits marker
REAL_T = 30.4            # magenta "real measurements" zone


def chart_alpha(t):
    t = loop_time(t)
    return smooth(21.2, 21.8, t) * (1 - smooth(REWIND_T0, REWIND_T0 + 0.6, t))


def hud_speed_text(t):
    return "REWIND" if rewinding(t) else f"×{SPEED:g}"


# ---------------------------------------------------------------- sound cue list (seconds, video time)
def cues():
    return {
        "release": RELEASE,
        "split_f32": T7,
        "split_f64": T16,
        "chart_in": 21.0,
        "points": list(zip(CHART_POINT_TIMES, [p[0] for p in CHART_POINTS])),
        "fit": (FIT_T0, FIT_T1),
        "minute": MINUTE_T,
        "real": REAL_T,
        "rewind": (REWIND_T0, REWIND_T1),
        "captions": [(a, b, " ".join(l)) for a, b, l in CAPTIONS],
    }


if __name__ == "__main__":
    print(f"T7 = {T7:.3f} s  T16 = {T16:.3f} s  speed x{SPEED}  tau at rewind = {TAU_REWIND0:.2f} s")
    for a, b, l in CAPTIONS:
        print(f"{a:6.2f}-{b:6.2f} ({b-a:4.2f} s) {' '.join(l)}  [{len(' '.join(l).split())} words]")
    for t in [0, 0.5, 1, 8, 18, 21, 33, 33.5, 34, 35, 35.5, 35.7, T_LAST]:
        print(f"t={t:7.3f} tau={sim_time(t):7.3f} zoom={zoom(t):.4f} chart={chart_mix(t):.2f}")
