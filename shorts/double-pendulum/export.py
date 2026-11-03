"""Run the 500-pendulum ensemble and export it for render.html.

sim.json : metadata + spread curves (60 Hz)
sim.bin  : float64 representative angles [S][2]  followed by
           float32 offsets from it            [S][2][N]
"""
import json, numpy as np
from sim import run, tip

TH1, TH2, DELTA, N = 140.0, 175.0, 1e-9, 500    # degrees
T = 22.9; RATE = 240; REP = 250    # covers the cold-open flash (to 22.7 s)

out, drift = run(TH1, TH2, DELTA, N, T, rate=RATE, sub=10)   # h = 1/2400 s
print("max energy drift (J):", drift)
S = out.shape[0]
rep = out[:, :, REP].copy()
d = (out - rep[:, :, None]).astype(np.float32)
with open("sim.bin", "wb") as f:
    f.write(rep.astype(np.float64).tobytes()); f.write(d.tobytes())

x, y = tip(out)
sep = np.hypot(x - x[:, REP:REP + 1], y - y[:, REP:REP + 1]).max(1)   # metres
env = np.maximum.accumulate(sep)
w = np.degrees(np.angle(np.exp(1j * (out[:, 1, :] - out[:, 1, REP:REP + 1]))))
deg = np.minimum(w.max(1) - w.min(1), 360.0)
k = np.arange(0, S, RATE // 60)
json.dump({"n": N, "samples": S, "rate": RATE, "rep": REP,
           "theta1": TH1, "theta2": TH2, "delta_deg": DELTA, "energy_drift": float(drift),
           "env": [float(v) for v in env[k]], "deg": [float(v) for v in deg[k]]}, open("sim.json", "w"))
for t in range(0, 23):
    print(t, "%.3e m  %.3e deg" % (env[t * RATE], deg[t * RATE]))
