"""Step 3: synthesize the full soundtrack in code, sample-accurately locked to the same video->sim clock (timeline.py)
and the saved simulation (data/sim.npz). Output: data/audio.wav (48 kHz, 24-bit, exactly 34.000 s) + data/audio_report.json

Layers
  bed      low ambient synth drone (A), ~9 dB under the effects
  lanes    one restless shimmer per lane, driven by its own pendulums' on-screen angular speed (left pan / right pan)
  split    soft detune "split" at each pair's measured separation moment
  merge    warm resolving chord + soft bloom at the measured re-merge
  rewind   reverse swoosh 31-33.85 s, then near-silence that flows into the start (seamless loop, no click)
"""
import json, os, sys
import numpy as np
from scipy import signal, ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "templates", "audio"))
import dsp  # noqa: E402  (repo's own numpy/scipy DSP kit: filters, reverb IR, BS.1770 loudness, true peak, wav io)
import timeline as TL  # noqa: E402

SR = dsp.SR
N = int(round(TL.DUR * SR))
t = np.arange(N) / SR
D = np.load(os.path.join(HERE, "data", "sim.npz"))
S, DT = D["S"], float(D["dt"])
MEAS = json.load(open(os.path.join(HERE, "data", "measure.json")))
R = dsp.rng(404)

# ---------------------------------------------------------------- control signals from the simulation
s = TL.sim_time(t)
f = np.clip(s / DT, 0, len(S) - 1.000001)
i = np.floor(f).astype(int)
a = (f - i)[:, None, None]
W = S[i, :, 2:] * (1 - a) + S[i + 1, :, 2:] * a          # (N, 4 pendulums, 2) angular velocities [rad/s sim]
TH = S[i, :, :2] * (1 - a) + S[i + 1, :, :2] * a
rate = np.where(t < TL.T_RW0, TL.rate(t), 0.0)          # forward playback rate; rewind handled separately
rate = np.maximum(rate, np.where(t >= TL.T_REL, 1.0, 0.0))
scr = np.abs(W).sum(axis=2) * rate[:, None]               # on-screen angular speed per pendulum
ctrl_lp = lambda x, fc=12.0: signal.sosfiltfilt(signal.butter(2, fc, fs=SR, output="sos"), x)
lane_speed = [ctrl_lp(scr[:, 0] + scr[:, 1]), ctrl_lp(scr[:, 2] + scr[:, 3])]
rw = TL.rewind_u(t)
not_rw = np.where(t < TL.T_RW0, 1.0, np.clip(1 - rw / 0.35, 0, 1))  # lane textures hand over to the rewind swoosh


def intensity(v, ref):
    return np.tanh(np.maximum(v, 0) / ref) ** 0.9


def env_ad(n, att, dec):
    tt = np.arange(n) / SR
    return np.minimum(tt / att, 1.0) * np.exp(-np.maximum(tt - att, 0) / dec)


def osc(freq_curve):
    return np.sin(2 * np.pi * np.cumsum(freq_curve) / SR)


# ---------------------------------------------------------------- bed: low ambient drone
def drone():
    out = np.zeros(N)
    lfo = dsp.smooth_noise(N, R, 0.15)
    for fr, g in ((55.0, 1.0), (82.41, 0.55), (110.0, 0.45), (164.81, 0.22), (220.0, 0.12)):
        for det in (-0.12, 0.12):
            ph = osc(np.full(N, fr * 2 ** (det / 12)) * (1 + 0.0015 * lfo))
            # soft saw-ish: a few harmonics
            out += g * (ph + 0.3 * np.sin(2 * np.arcsin(np.clip(ph, -1, 1))))
    out = dsp.lp(out, 520, 2)
    out = dsp.hp(out, 40, 4)
    sw = 0.85 + 0.15 * np.sin(2 * np.pi * t / 8.5)
    return dsp.widen(out * sw, R, 13.0, 0.7)


# ---------------------------------------------------------------- per-lane shimmer textures
def shimmer(lane):
    v = lane_speed[lane]
    I = intensity(v, 14.0 if lane == 0 else 10.0) * not_rw
    base = [880.0, 1318.5, 1760.0, 2217.5] if lane == 0 else [440.0, 659.3, 880.0, 1108.7]
    p = TL  # noqa
    pen = 0 if lane == 0 else 2
    wob = np.tanh(ctrl_lp(W[:, pen, 1] * np.maximum(rate, 1), 6.0) / 12.0)   # bottom-arm swing direction/speed
    out = np.zeros(N)
    for k, fb in enumerate(base):
        trem = 0.55 + 0.45 * np.sin(2 * np.pi * (5.5 + 2.7 * k) * t + k) * (0.4 + 0.6 * I)
        fc = fb * (1 + 0.035 * wob * (1 if k % 2 else -1) + 0.004 * dsp.smooth_noise(N, R, 3.0))
        out += (0.8 ** k) * trem * osc(fc)
    # grainy air: band-passed noise, gated by fast speed fluctuations
    nz = dsp.bp(R.standard_normal(N), 1800 if lane == 0 else 900, 5200 if lane == 0 else 2600, 2)
    gate = np.clip(0.5 + 0.5 * dsp.smooth_noise(N, R, 18.0), 0, 1)
    out = out * 0.6 + nz * gate * 0.9
    out = dsp.lp(out * I, 7000, 2)
    return dsp.pan(out, -0.65 if lane == 0 else 0.65)


# ---------------------------------------------------------------- one-shots
def split_sound(t0, lane):
    n = dsp.ns(1.6)
    tt = np.arange(n) / SR
    f0 = 1046.5 if lane == 0 else 784.0
    det = 1 - np.exp(-tt / 0.18)                              # quick detune apart (+-45 cents)
    x = np.zeros(n)
    for sgn in (-1, 1):
        x += osc(f0 * 2 ** (sgn * 0.45 * det / 12)) + 0.35 * osc(2 * f0 * 2 ** (sgn * 0.45 * det / 12))
    x *= env_ad(n, 0.015, 0.45)
    return dsp.pan(dsp.lp(x, 6000), -0.55 if lane == 0 else 0.55), t0


def merge_chord(t0):
    n = dsp.ns(4.5)
    tt = np.arange(n) / SR
    notes = [110.0, 164.81, 220.0, 277.18, 329.63, 493.88]   # A major add9 (resolves the A drone)
    x = np.zeros(n)
    for k, fr in enumerate(notes):
        for det in (-0.05, 0.05):
            x += (0.9 ** k) * osc(np.full(n, fr * 2 ** (det / 12))) * (1 + 0.15 * np.sin(2 * np.pi * 0.5 * tt + k))
    env = np.minimum(tt / 0.25, 1.0) ** 2 * np.exp(-np.maximum(tt - 0.25, 0) / 1.6)
    x = dsp.lp(x * env, 2600)
    bloom = dsp.bp(R.standard_normal(n), 300, 3000) * env_ad(n, 0.12, 0.6) * 0.25
    return dsp.widen(x + bloom, R, 9.0, 0.8), t0 - 0.06


def rewind_swoosh():
    n = N - dsp.ns(TL.T_RW0)
    tt = np.arange(n) / SR
    T = TL.T_RW1 - TL.T_RW0
    u = np.clip(tt / T, 0, 1)
    sp = np.gradient(TL.rewind_ease(u)) * SR                 # rewind speed profile (eases in and out)
    sp /= sp.max()
    fc = 250 * (14 ** sp)                                    # filter sweeps with rewind speed
    nz = R.standard_normal(n)
    y = np.zeros(n)
    for fb in np.geomspace(200, 5000, 10):                   # fixed band-pass bank, crossfaded by the sweep
        band = dsp.bp(nz, fb / 1.35, fb * 1.35, 2)
        y += band * np.exp(-0.5 * (np.log(fc / fb) / 0.35) ** 2)
    # reverse-shaped envelope: swells toward the rush, then falls smoothly to near-silence by the loop point
    env = sp ** 1.3 * np.clip((T + 0.05 - tt) / 0.5, 0, 1)
    chirp = osc(300 + 1500 * sp) * sp ** 2 * 0.25            # tape-rewind whirr
    y = dsp.lp(y * env + chirp * np.clip((T - tt) / 0.4, 0, 1), 6000)
    return dsp.widen(y, R, 11.0, 0.9)


# ---------------------------------------------------------------- mix
def limiter(x, ceil_db=-1.5, look=0.003, rel=0.08):
    c = dsp.dbg(ceil_db)
    n = x.shape[1]
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(0)
    pk = np.pad(up, (0, max(0, 4 * n - len(up))))[: 4 * n].reshape(n, 4).max(1)
    gr = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    La = max(1, dsp.ns(look))
    g = ndimage.minimum_filter1d(gr, 2 * La + 1, mode="nearest")
    # release: one-pole toward 1 (vectorised via exponential min-filter approximation)
    kr = np.exp(-1 / (rel * SR))
    gg = signal.lfilter([1 - kr], [1, -kr], g - 1.0) + 1.0   # smooth
    gg = np.minimum(gg, g)                                    # never exceed the required reduction
    gg = ndimage.uniform_filter1d(gg, 2 * La + 1, mode="nearest")
    return x * np.minimum(gg, 1.0)[None, :], float(gg.min())


def main():
    fx = np.zeros((2, N))
    bed = drone()
    lanes = shimmer(0) * 1.0 + shimmer(1) * 1.0
    fx += lanes
    cues = []
    for lane, key in ((0, "split_vis_left_video"), (1, "split_vis_right_video")):
        x, t0 = split_sound(MEAS[key], lane)
        dsp.place(fx, x * 0.9, t0)
        cues.append({"cue": f"split_{'left' if lane == 0 else 'right'}", "video_t": round(t0, 4)})
    x, t0 = merge_chord(MEAS["remerge_right_video"])
    dsp.place(fx, x * 1.1, t0)
    cues.append({"cue": "merge_chord", "video_t": round(t0, 4)})
    sw = rewind_swoosh()
    dsp.place(fx, sw * 2.0, TL.T_RW0)
    cues.append({"cue": "rewind_swoosh", "video_t": TL.T_RW0, "end": TL.T_RW1})
    # bed level: ~9 dB under the effects (integrated, measured over the forward part)
    fwd = slice(dsp.ns(0.6), dsp.ns(31.0))
    g_bed = dsp.dbg(dsp.integrated(fx[:, fwd]) - 9.0 - dsp.integrated(bed[:, fwd]))
    bed *= g_bed
    # bed ducks out with the rewind and fades in over the opening (near-silence at the loop point)
    bed_env = np.clip(t / 0.45, 0, 1) ** 2 * np.clip((TL.T_RW1 + 0.1 - t) / 2.2, 0.0, 1.0)
    mix = fx + bed * bed_env[None, :]
    # room
    ir = dsp.make_ir(1.8, 7, hf=1.2)
    mix = mix + 0.18 * dsp.convolve_st(mix.mean(0), ir)
    mix = dsp.hp(mix, 40, 4)                                 # phone friendly: nothing below ~40 Hz
    mix = dsp.shelf(mix, 8000, -3.0, "high")                 # no harsh highs
    # loop point: both ends glide to silence (no click when the video loops)
    edge = np.ones(N)
    edge[:dsp.ns(0.03)] = dsp.ramp(dsp.ns(0.03))
    edge[-dsp.ns(0.12):] = dsp.ramp(dsp.ns(0.12))[::-1]
    mix *= edge[None, :]
    # loudness -14 LUFS, true peak <= -1 dBTP (limit to -1.5 for AAC headroom)
    for _ in range(4):
        mix *= dsp.dbg(-14.0 - dsp.integrated(mix))
        mix, gmin = limiter(mix, -1.6)
    mix *= dsp.dbg(-14.0 - dsp.integrated(mix))
    rep = {"integrated_lufs": dsp.integrated(mix), "true_peak_dbtp": dsp.true_peak(mix), "samples": N, "sr": SR,
           "bed_vs_fx_db": -9.0, "start_abs_max": float(np.abs(mix[:, :48]).max()),
           "end_abs_max": float(np.abs(mix[:, -48:]).max()), "cues": cues}
    dsp.write_wav(os.path.join(HERE, "data", "audio.wav"), mix, 24)
    json.dump(rep, open(os.path.join(HERE, "data", "audio_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
