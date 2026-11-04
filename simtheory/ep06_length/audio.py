"""SimTheory #06 soundtrack: synthesized in code at 48 kHz, sample-accurate to timeline.py (same clock as the picture).
Layers: ambient drone bed (~9 dB under the effects), four speed-driven pendulum textures, flip pops,
merge chord, rising bar ticks, reverse swoosh into near-silence at the loop point. Output: out/audio.wav
"""
import json, math, os, sys
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "templates", "audio"))
import dsp  # noqa: E402  (repo DSP kit: filters, BS.1770 loudness, true peak, wav io)
import timeline as TL  # noqa: E402

SR = dsp.SR
N = int(TL.DUR * SR)
t = np.arange(N) / SR
PHYS, SIM = TL.load()
LABELS = list(PHYS["pendulums"].keys())
TRAJ, LEN, HZ = SIM["traj"], SIM["L"], int(SIM["save_hz"])
R = np.random.default_rng(606)

# chord the four textures resolve into (D major add9 spread): 1:3 D3, 1:1 A3, 3:1 E4, 9:1 F#4
NOTE = {"1:3": 146.83, "1:1": 220.00, "3:1": 329.63, "9:1": 369.99}
PAN = {"1:3": -0.55, "1:1": 0.55, "3:1": -0.4, "9:1": 0.4}

# ---------------------------------------------------------------- clock
s_vid = np.array([TL.sim_time(x) for x in t])                    # sim time at every audio sample
ds_dt = np.gradient(s_vid) * SR                                   # playback rate (1 forward, large negative in rewind)


def bob2_speed(k, s):
    x = s * HZ
    i0 = np.clip(np.floor(x).astype(int), 0, TRAJ.shape[1] - 2)
    f = x - i0
    st = TRAJ[k, i0] * (1 - f)[:, None] + TRAJ[k, i0 + 1] * f[:, None]
    L1, L2 = LEN[k]
    th1, th2, w1, w2 = st.T
    vx = L1 * w1 * np.cos(th1) + L2 * w2 * np.cos(th2)
    vy = L1 * w1 * np.sin(th1) + L2 * w2 * np.sin(th2)
    return np.hypot(vx, vy)


def env_lin(pts):
    """piecewise-linear envelope over the whole timeline from [(time, value), ...]"""
    xs, ys = zip(*pts)
    return np.interp(t, xs, ys)


def smoothstep_env(a, b):
    u = np.clip((t - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def onepole(x, hz):
    a = math.exp(-2 * math.pi * hz / SR)
    return signal.lfilter([1 - a], [1, -a], x)


# merge weight: 0 = textures follow their bob's speed, 1 = resolved into the chord
merge = smoothstep_env(TL.REVEAL, TL.REVEAL + 1.6) * (1 - smoothstep_env(TL.REWIND, TL.REWIND + 0.6))

# ---------------------------------------------------------------- pendulum textures
tex = np.zeros((2, N))
for k, lab in enumerate(LABELS):
    v = bob2_speed(k, s_vid) * np.abs(ds_dt)                      # perceived speed of the bob on screen
    v = onepole(np.minimum(v, 60.0), 30.0)
    vn = np.clip(v / 8.0, 0, 3.0)                                 # ~1 at the fastest real-time speed
    base = NOTE[lab]
    f_speed = base * 2 ** (0.9 * np.tanh(vn))                    # up to ~an octave higher when fast
    f = np.exp((1 - merge) * np.log(f_speed) + merge * np.log(base))
    bright = (1 - merge) * (0.15 + 0.85 * np.tanh(1.3 * vn)) + merge * 0.55
    amp_speed = 0.12 + 0.88 * np.tanh(1.6 * vn)
    amp = (1 - merge) * amp_speed + merge * 0.75
    y = np.zeros(N)
    for det in (-0.004, 0.0, 0.004):                              # gentle chorus
        ph = dsp.phase(f * (1 + det)) + R.uniform(0, 2 * np.pi)
        for n in range(1, 9):
            if base * n * 2 > 9000:
                break
            w = (1.0 / n) * np.exp(-(n - 1) / (0.6 + 5.0 * bright))
            y += w * np.sin(n * ph)
    y *= amp / 3.0
    tex += dsp.pan(y, PAN[lab]) * 0.5

# merge chord bloom: octave + fifth pad that swells as the four drawings merge
pad = np.zeros(N)
for fr, a in [(73.42, 0.5), (146.83, 0.35), (220.0, 0.3), (293.66, 0.25), (369.99, 0.2), (440.0, 0.15), (587.33, 0.08)]:
    ph = dsp.phase(np.full(N, fr) * (1 + 0.002 * np.sin(2 * np.pi * 0.13 * t + fr)))
    pad += a * (np.sin(ph) + 0.25 * np.sin(2 * ph))
pad_env = smoothstep_env(TL.REVEAL + 0.4, TL.REVEAL + 2.0) * (1 - 0.35 * smoothstep_env(TL.CHART, TL.CHART + 0.8)) \
    * (1 - smoothstep_env(TL.REWIND, TL.REWIND + 0.5))
pad = dsp.lp(pad * pad_env, 2500)
fx = tex + dsp.widen(pad, R, amount=0.6) * 0.45

# ---------------------------------------------------------------- one-shots
cues = []


def blip(f0, f1, dur, decay):
    tt = np.arange(int(dur * SR)) / SR
    f = f1 + (f0 - f1) * np.exp(-tt / (dur * 0.25))
    y = np.sin(dsp.phase(f)) * np.exp(-tt / decay)
    a = min(len(y), int(0.002 * SR))
    y[:a] *= np.linspace(0, 1, a)
    return y


# flip pops: small, pitched per pendulum (two octaves over its note)
for tv, k, lab in TL.flip_events(PHYS):
    y = blip(NOTE[lab] * 4 * 1.25, NOTE[lab] * 4, 0.12, 0.035)
    y[: int(0.05 * SR)] += 0.3 * blip(NOTE[lab] * 8, NOTE[lab] * 8, 0.05, 0.01)
    dsp.place(fx, dsp.pan(y, PAN[lab]), tv, 0.55)
    cues.append(("flip_pop", round(tv, 4), lab))

# release: soft low thump + air as the pendulums drop
th = blip(110, 55, 0.5, 0.12)
air = dsp.bp(R.standard_normal(int(0.8 * SR)), 600, 4000) * np.exp(-np.arange(int(0.8 * SR)) / SR / 0.25) * 0.15
dsp.place(fx, th * 0.8, TL.RELEASE)
dsp.place(fx, dsp.widen(air, R), TL.RELEASE)
cues.append(("release", TL.RELEASE, "all"))

# light-painting starts: shimmer
sh = np.zeros(int(1.2 * SR)); ts = np.arange(len(sh)) / SR
for fr in (1174.7, 1479.98, 1760.0, 2349.3):
    sh += np.sin(2 * np.pi * fr * ts + R.uniform(0, 6)) * np.exp(-ts / 0.45)
sh *= np.minimum(ts / 0.03, 1)
dsp.place(fx, dsp.widen(sh * 0.06, R, amount=0.8), TL.TRAIL_ON)
cues.append(("trails_start_shimmer", TL.TRAIL_ON, "all"))

# merge: soft swell already in pad; add a gentle lift whoosh
nw = int(1.6 * SR); tw = np.arange(nw) / SR
wh = dsp.bp(R.standard_normal(nw), 300, 3000) * np.sin(np.pi * tw / 1.6) ** 2 * 0.12
dsp.place(fx, dsp.widen(wh, R), TL.REVEAL)
cues.append(("merge_whoosh_chord", TL.REVEAL, "all"))

# chart: soft rising tick as each bar grows
for j, row in enumerate(TL.chart_rows(PHYS)):
    f0 = 660 * 2 ** (j * 4 / 12)
    # rising: start low, glide up
    tt2 = np.arange(int(0.16 * SR)) / SR
    f = f0 * (1 + 0.5 * (tt2 / 0.16) ** 0.7)
    y = np.sin(dsp.phase(f)) * np.sin(np.pi * np.minimum(tt2 / 0.16, 1)) ** 1.5 * np.exp(-tt2 / 0.12)
    dsp.place(fx, dsp.pan(y * 0.35, -0.2 + 0.13 * j), row["t0"])
    cues.append(("bar_tick", round(row["t0"], 4), row["label"]))

# result accents (soft) under the two result captions
for tc, lab in ((23.5, TL.ranking(PHYS)[0]), (26.0, TL.ranking(PHYS)[-1])):
    y = blip(NOTE[lab] * 2, NOTE[lab] * 2, 0.9, 0.3) * 0.25 + blip(NOTE[lab] * 3, NOTE[lab] * 3, 0.9, 0.2)[: int(0.9 * SR)] * 0.1
    dsp.place(fx, dsp.widen(y, R), tc)
    cues.append(("result_accent", tc, lab))

# rewind: reverse swoosh (a whoosh played backwards: swells and cuts), ending in near-silence
nr = int(2.6 * SR); tr = np.arange(nr) / SR
nz = R.standard_normal(nr)
fwd = np.zeros(nr)
for lo, hi, g in ((200, 1200, 1.0), (1200, 5000, 0.5)):
    fwd += g * dsp.bp(nz, lo, hi)
fwd *= np.exp(-tr / 0.55) * np.minimum(tr / 0.01, 1)
rev = fwd[::-1].copy()
rev *= 1 - np.clip((tr - 2.45) / 0.15, 0, 1)                    # fade the very end instead of a hard cut
dsp.place(fx, dsp.widen(rev * 0.35, R, amount=0.8), TL.REWIND)
cues.append(("reverse_swoosh", TL.REWIND, "all"))

# ---------------------------------------------------------------- drone bed
bed = np.zeros(N)
for fr, a in ((73.42, 1.0), (110.0, 0.55), (146.83, 0.35), (220.0, 0.12)):
    ph = dsp.phase(np.full(N, fr) * (1 + 0.0015 * np.sin(2 * np.pi * 0.07 * t + fr)))
    bed += a * (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
bed *= 0.75 + 0.25 * np.sin(2 * np.pi * t / 8.0) ** 2             # slow breathing (period 8 s, loops: 32/8 = 4)
bed = dsp.lp(bed, 900) + 0.08 * dsp.bp(R.standard_normal(N), 400, 2500)
bed = dsp.widen(bed, R, amount=0.7)

# ---------------------------------------------------------------- mix
loop_env = env_lin([(0, 0.0), (0.35, 1.0), (31.2, 1.0), (31.9, 0.0), (32, 0.0)])   # near-silence at the loop seam
fx_l = dsp.integrated(fx * loop_env)
bed_l = dsp.integrated(bed * loop_env)
bed_gain = dsp.dbg((fx_l - 9.0) - bed_l)                          # bed sits 9 dB under the effects
mix = (fx + bed * bed_gain) * loop_env
mix = dsp.hp(mix, 40, order=4)                                    # phone-friendly: nothing below ~40 Hz
mix = dsp.lp(mix, 12000, order=2)                                 # no harsh highs
mix = dsp.shelf(mix, 6000, -2.0, "high")


def limiter(x, ceiling_db=-1.3, look=0.004, release=0.08):
    """look-ahead peak limiter driven by 4x oversampled (true-peak) detection"""
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(0)
    pk = up.reshape(-1, 4).max(1)[: x.shape[1]]
    ceil = dsp.dbg(ceiling_db)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = int(look * SR)
    from scipy.ndimage import minimum_filter1d
    gmin = minimum_filter1d(g, size=2 * la + 1)                    # look-ahead window around each sample
    a = math.exp(-1 / (release * SR))
    out = np.empty_like(gmin); cur = 1.0
    for i in range(len(gmin)):
        cur = gmin[i] if gmin[i] < cur else a * cur + (1 - a) * gmin[i]
        out[i] = cur
    out = np.minimum(onepole(out, 400.0), out)                   # smooth the attack corners, never above target
    return x * out


target = -14.0
for it in range(4):
    mix = mix * dsp.dbg(target - dsp.integrated(mix))
    mix = limiter(mix)
    print(f"iter {it}: integrated {dsp.integrated(mix):.2f} LUFS, true peak {dsp.true_peak(mix):.2f} dBTP")
mix *= dsp.dbg(target - dsp.integrated(mix))
if dsp.true_peak(mix) > -1.05:
    mix *= dsp.dbg(-1.05 - dsp.true_peak(mix))
# seam: first/last samples exactly zero with short fades (no click at the loop point)
fl = int(0.01 * SR)
mix[:, :fl] *= np.linspace(0, 1, fl)
mix[:, -fl:] *= np.linspace(1, 0, fl)
os.makedirs("out", exist_ok=True)
dsp.write_wav("out/audio.wav", mix, bits=24)
rep = dict(integrated_lufs_internal=round(float(dsp.integrated(mix)), 2), true_peak_dbtp_internal=round(float(dsp.true_peak(mix)), 2),
           fx_vs_bed_db=9.0, bed_gain_db=round(20 * math.log10(bed_gain), 2),
           seam_first_sample=float(np.abs(mix[:, 0]).max()), seam_last_sample=float(np.abs(mix[:, -1]).max()),
           seam_rms_last_50ms_dbfs=round(20 * math.log10(np.sqrt((mix[:, -2400:] ** 2).mean()) + 1e-12), 1),
           seam_rms_first_50ms_dbfs=round(20 * math.log10(np.sqrt((mix[:, :2400] ** 2).mean()) + 1e-12), 1),
           cues=cues)
json.dump(rep, open("data/audio_report.json", "w"), indent=1)
print({k: v for k, v in rep.items() if k != "cues"})
