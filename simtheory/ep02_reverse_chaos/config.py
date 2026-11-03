"""Locked tuning for SimTheory #02 (chosen from sweep in tune.py; see LOG.md)."""
TH0_DEG = 117.0          # theta1 = theta2 at release, both at rest (tuned from 120, see LOG.md)
T_SIM = 24.5             # forward run length [s]
NUDGE_DEG = 1e-10        # ORANGE: theta1 changed by 0.0000000001 deg at the reversal
DIVERGE_DEG = 2.0        # "visibly separated" threshold
FPS = 60
W, H = 1440, 2560
DUR = 30.0
