"""Shared constants for SimTheory #05 - CAN A PENDULUM FLIP FOREVER?"""
import numpy as np

# ---------------------------------------------------------------- video
W, H = 1440, 2560
FPS = 60
DUR = 42.0
NFRAMES = 2520
SR = 48000

# safe area for all text / charts
SAFE = dict(x0=86, x1=1152, y0=384, y1=1664)

# ---------------------------------------------------------------- physics
G = 9.81
L = 1.0
M = 1.0
W0 = np.sqrt(G / L)               # omega_0 = sqrt(g/L)
WC = 2.0 * W0                     # critical bottom speed (just reaches the top)
EC = 2.0 * M * G * L              # critical energy
FRACS = [0.2, 0.4, 0.6, 0.8, 0.95, 1.00, 1.05, 1.3, 1.7]
THR = FRACS.index(1.00)           # index of the analytic threshold pendulum
NUM = [i for i in range(len(FRACS)) if i != THR]
PUSH = [WC * np.sqrt(f) for f in FRACS]

DT = 1.0 / 4800.0                 # 0.2083 ms  (80 steps per 60-fps frame)
T_SIM = 40.0                      # simulated span (video uses up to 39.5 s)
T_PUSH = 0.5                      # video time of the push

# ---------------------------------------------------------------- layout
PIVOT = np.array([(SAFE["x0"] + SAFE["x1"]) / 2.0, (SAFE["y0"] + SAFE["y1"]) / 2.0])  # (619, 1024)
R = 570.0                         # arm length in px (full circle stays in frame, bob path clear of legend)
TOP = PIVOT + np.array([0.0, -R])

# ---------------------------------------------------------------- palette
def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0

CYAN, BLUE, PURPLE, MAGENTA, ORANGE, WHITE, GOLD = map(hexc, ["22E5FF", "3D7BFF", "9B5CFF", "FF4FD8", "FF8A2A", "FFFFFF", "FFC94A"])

def mix(a, b, s):
    return a * (1 - s) + b * s

COLORS = [BLUE, mix(BLUE, CYAN, 0.5), CYAN, mix(CYAN, PURPLE, 0.5), PURPLE, WHITE,
          MAGENTA, mix(MAGENTA, ORANGE, 0.5), ORANGE]

FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_REG = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FONT_LIGHT = "/usr/share/fonts/opentype/inter/Inter-Light.otf"
FONT_MONO = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf"

HOOK = "Can a pendulum flip forever?"
