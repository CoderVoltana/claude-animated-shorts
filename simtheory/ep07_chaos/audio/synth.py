"""Stage 4: synthesize the whole soundtrack in code, sample-accurately locked to timeline.py.

  - low ambient drone bed (no music track attached), set ~9 dB under the effects
  - three voices (orange / cyan / white copies) in unison; each detunes when ITS copy's measured
    angle error crosses 1 deg (driven from the saved sim data through the video->sim mapping)
  - chimes for chart points (pitch rises with digits), swell for the fitted line
  - reverse swoosh for the rewind, ending in near-silence that flows into the start

Writes out/audio/{bed,fx,mix_raw}.wav (48 kHz float). Loudness is set by master.py.
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import timeline as T  # noqa: E402

SR = 48000
N = int(round(36.0 * SR))
t = np.arange(N) / SR
OUT = os.path.join(ROOT, "out", "audio")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- control signals at 1 kHz
CR = 1000
tc = np.arange(int(36.0 * CR) + 1) / CR
tau_c = np.array([T.sim_time(x) for x in tc])
runs = {k: np.load(os.path.join(T.DATA, f"run_{k}.npy"))[:, 1] for k in ["f32", "f64", "mp100"]}


def theta2(name, tau):
    a = runs[name]
    x = np.clip(tau / T.M["dt"], 0, len(a) - 1.000001)
    i = np.floor(x).astype(int)
    f = x - i
    return a[i] * (1 - f) + a[i + 1] * f


def err_deg(name, tau):
    d = np.degrees(theta2(name, tau) - theta2("mp100", tau))
    return np.abs((d + 180) % 360 - 180)


def up(ctrl):
    """control-rate -> audio-rate (linear)"""
    return np.interp(t, tc, ctrl)


def sm(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0, 1)
    return u * u * u * (u * (u * 6 - 15) + 10)


def env_adsr(n, a, d, sus_level=0.0):
    out = np.ones(n)
    na = max(1, int(a * SR))
    out[:na] = np.linspace(0, 1, na)
    out[na:] = np.exp(-np.arange(n - na) / (d * SR)) * (1 - sus_level) + sus_level
    return out


def place(buf, sig, t0, gain=1.0, pan=0.0):
    i0 = int(round(t0 * SR))
    i1 = min(N, i0 + len(sig))
    if i1 <= i0:
        return
    gl, gr = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[i0:i1, 0] += gain * gl * sig[: i1 - i0]
    buf[i0:i1, 1] += gain * gr * sig[: i1 - i0]


# ---------------------------------------------------------------- global envelope: near-silence at both ends
# start: quick rise from near silence; end: settles to near silence (identical level at the seam)
edge = np.clip(sm(0.0, 0.35, t) * (1 - sm(35.25, 35.95, t)), 0, 1)
edge = 0.003 + 0.997 * edge
edge[0] = edge[-1] = 0.0
edge[:48] *= np.linspace(0, 1, 48)        # 1 ms guards: exact zero at the loop seam, no click
edge[-48:] *= np.linspace(1, 0, 48)

# chart phase: voices step back while the pendulums are a small inset
chart_c = np.array([T.chart_mix(x) for x in tc])
chart = up(chart_c)

# ---------------------------------------------------------------- bed: low ambient drone
bed = np.zeros((N, 2))
lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / 9.0)
for f0, amp, det in ((55.0, 0.55, 0.15), (82.41, 0.35, 0.12), (110.0, 0.30, 0.2), (164.81, 0.12, 0.25), (220.0, 0.06, 0.3)):
    for ch, s in ((0, -1), (1, 1)):
        ph = 2 * np.pi * (f0 + s * det) * t + rng.uniform(0, 2 * np.pi)
        tone = np.sin(ph) + 0.25 * np.sin(2 * ph) * lfo + 0.08 * np.sin(3 * ph)
        bed[:, ch] += amp * tone
air = sosfiltfilt(butter(2, [300, 1400], "bandpass", fs=SR, output="sos"), rng.standard_normal((N, 2)), axis=0)
bed += 0.04 * air * (0.6 + 0.4 * lfo)[:, None]
bed *= (0.85 + 0.15 * chart)[:, None] * edge[:, None]

# ---------------------------------------------------------------- the three voices
fx = np.zeros((N, 2))
BASE = 220.0  # A3
voice_env = sm(0.0, 0.45, t) * (1 - 0.68 * chart)
rw_speed = np.abs(np.gradient(tau_c, 1 / CR))                    # sim-s per video-s
rewind_c = np.array([1.0 if T.rewinding(x) else 0.0 for x in tc])
tape = up(np.tanh(np.maximum(rw_speed - T.SPEED * 1.05, 0) / 12) * rewind_c) * 380   # tape-rewind pitch rise (cents)


def detune_cents(name, sign, depth):
    e = err_deg(name, tau_c)
    # zero until the measured error crosses 1 deg (the horizon), then glide into dissonance
    x = np.clip(np.log10(np.maximum(e, 1e-12)) / 1.3, 0, 1)        # 1 deg -> 0, 20 deg -> 1
    c = depth * (x * x * (3 - 2 * x))
    wob = 18 * np.sin(np.radians(e) * 3) * (x > 0)                  # keeps moving with the chaotic error
    return sign * (c + wob)


def voice(cents_c, pan, amp, bright):
    f = BASE * 2 ** ((up(cents_c) + tape) / 1200)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = (np.sin(ph) + bright * 0.42 * np.sin(2 * ph) + bright * 0.20 * np.sin(3 * ph) + bright * 0.08 * np.sin(4 * ph))
    trem = 1 + 0.04 * np.sin(2 * np.pi * 0.23 * t + pan)
    out = np.zeros((N, 2))
    out[:, 0] = s * np.sqrt(0.5 * (1 - pan))
    out[:, 1] = s * np.sqrt(0.5 * (1 + pan))
    return out * (amp * trem * voice_env)[:, None]


zeros = np.zeros_like(tc)
v_orange = voice(detune_cents("f32", +1, 85), -0.35, 0.30, 1.0)
v_cyan = voice(detune_cents("f64", -1, 70), 0.35, 0.30, 1.0)
v_white = voice(zeros, 0.0, 0.30, 1.0)
fx += v_orange + v_cyan + v_white
voices = v_orange + v_cyan + v_white

# split accents: soft low "pluck" + short airy ring exactly at each split
for ts, pan in ((T.T7, -0.35), (T.T16, 0.35)):
    n = int(1.2 * SR)
    tt = np.arange(n) / SR
    pl = np.sin(2 * np.pi * 110 * tt) * np.exp(-tt / 0.35) + 0.4 * np.sin(2 * np.pi * 330 * tt) * np.exp(-tt / 0.2)
    pl *= np.minimum(1, tt / 0.004)
    place(fx, pl, ts, 0.35, pan)
    nz = sosfilt(butter(2, [1500, 5000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    place(fx, nz * np.exp(-tt / 0.25) * np.minimum(1, tt / 0.02), ts, 0.05, pan)


# ---------------------------------------------------------------- chart chimes (pitch rises with digits)
def bell(f, dur=2.2, amp=1.0):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.9)
         + 0.35 * np.sin(2 * np.pi * f * 2.0 * tt) * np.exp(-tt / 0.45)
         + 0.12 * np.sin(2 * np.pi * f * 3.01 * tt) * np.exp(-tt / 0.22))
    return amp * s * np.minimum(1, tt / 0.003)


CHIME_NOTES = {7: 523.25, 16: 659.26, 24: 783.99, 32: 880.00, 40: 1046.50}   # C5 E5 G5 A5 C6
for tp, d in zip(T.CHART_POINT_TIMES, [p[0] for p in T.CHART_POINTS]):
    place(fx, bell(CHIME_NOTES[d]), tp, 0.30, -0.3 + 0.6 * (d - 7) / 33)

# fitted-line swell: soft pad chord + filtered noise riser, peaking as the line finishes
t0, t1 = T.FIT_T0 - 0.2, T.FIT_T1
n = int((t1 - t0 + 2.0) * SR)
tt = np.arange(n) / SR
sw_env = np.where(tt < (t1 - t0), sm(0, t1 - t0, tt) ** 1.5, np.exp(-(tt - (t1 - t0)) / 0.7))
pad = sum(a * np.sin(2 * np.pi * f * tt + p) for f, a, p in ((220, 0.5, 0), (277.18, 0.35, 1), (329.63, 0.35, 2), (493.88, 0.18, 3)))
noise = sosfilt(butter(2, [600, 3000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
place(fx, (pad * 0.6 + noise * 0.25) * sw_env, t0, 0.32)
# gold minute marker: two-note chime
place(fx, bell(1046.5, 1.6) + 0.7 * np.concatenate([np.zeros(int(0.12 * SR)), bell(1318.5, 1.6)])[: int(1.6 * SR)],
      T.MINUTE_T + 0.3, 0.18)
# "real measurements" zone: a soft low glow tone
n = int(1.8 * SR)
tt = np.arange(n) / SR
place(fx, (np.sin(2 * np.pi * 146.83 * tt) + 0.3 * np.sin(2 * np.pi * 293.66 * tt)) * sm(0, 0.15, tt) * np.exp(-tt / 0.6),
      T.REAL_T, 0.22)

# ---------------------------------------------------------------- rewind: quick reverse swoosh
r0, r1 = T.REWIND_T0, T.REWIND_T1
n = int((r1 - r0) * SR)
tt = np.arange(n) / SR
nz = rng.standard_normal(n)
# swept band: a forward swoosh (bright->dark) played backwards = rising, sucking-in sound
fc = np.exp(np.linspace(np.log(4500), np.log(350), n))[::-1]
sw = np.zeros(n)
blk = 1024
for i in range(0, n, blk):
    f = fc[i:i + blk].mean()
    sos = butter(2, [f * 0.6, min(f * 1.6, 9000)], "bandpass", fs=SR, output="sos")
    sw[i:i + blk] = sosfilt(sos, nz[i:i + blk])
fwd_env = np.exp(-tt / 0.6) * np.minimum(1, tt / 0.01)
rev_env = fwd_env[::-1]
rev_env *= 1 - sm(n / SR - 0.25, n / SR, tt)          # ends quietly, no hard stop
place(fx, sw * rev_env, r0, 0.55)
place(fx, sw[::-1] * np.exp(-tt / 0.25) * np.minimum(1, tt / 0.01), r0, 0.18)   # tape-stop grab at the start

fx *= edge[:, None]

# ---------------------------------------------------------------- write stems (master.py balances & normalises)
for name, buf in (("bed", bed), ("fx", fx), ("voices", voices * edge[:, None])):
    wavfile.write(os.path.join(OUT, f"{name}.wav"), SR, buf.astype(np.float32))
print("stems written", N, "samples")
