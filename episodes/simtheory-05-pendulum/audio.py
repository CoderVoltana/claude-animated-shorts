"""Step 4: synthesize the whole soundtrack in code, sample-accurately synced to the video timeline (48 kHz).

All cue times come from timeline.py (video-time <-> sim-time mapping) and data/events_sim.json (bottom passes
measured from the simulation). Writes out/mix.wav (stereo, 42.000 s exactly) and data/audio_cues.json.

Loop-safety: the drone uses only frequencies on the 1/42 Hz grid (exactly periodic over 42 s); filters and reverb
are applied circularly (3 tiled copies, middle kept), and every one-shot sound has died out before 41.75 s.
"""
import json
import math
import os

import numpy as np
import pyloudnorm as pyln
from scipy import signal

import timeline as TL
from config import DUR, FRACS, SR, THR, W0

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(round(DUR * SR))
TAU = 2 * np.pi
rng = np.random.default_rng(5)

NOTES = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.26, 783.99]   # C major pentatonic, by energy


def q42(f):
    """Snap a frequency onto the 1/42 Hz grid so the tone is exactly periodic over the 42 s loop."""
    return round(f * DUR) / DUR


def circ_filter(sos, x):
    y = signal.sosfilt(sos, np.concatenate([x, x, x], axis=-1), axis=-1)
    return y[..., N:2 * N]


def circ_conv(x, ir):
    L = 1 << int(np.ceil(np.log2(N + len(ir))))
    X = np.fft.rfft(x, L)
    y = np.fft.irfft(X * np.fft.rfft(ir, L), L)
    out = y[:N].copy()
    out[:len(y) - N] += y[N:N + (len(y) - N)][:N] if len(y) > N else 0
    return out


def pan(mono, p):
    """p in [-1, 1], equal-power."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)])


def add(buf, start_s, sig):
    i0 = int(round(start_s * SR))
    if i0 >= N:
        return
    if i0 < 0:
        sig = sig[..., -i0:]
        i0 = 0
    n = min(sig.shape[-1], N - i0)
    buf[..., i0:i0 + n] += sig[..., :n]


# ------------------------------------------------------------------ instruments
def ping(f, amp, dur=None):
    dur = dur or (1.6 - 0.6 * (f - 260) / 520)
    t = np.arange(int(dur * SR)) / SR
    dec = 0.55 - 0.25 * (f - 260) / 520
    env = (1 - np.exp(-t / 0.004)) * np.exp(-t / dec)
    s = (np.sin(TAU * f * t) + 0.18 * np.sin(TAU * 2 * f * t) * np.exp(-t / (dec * 0.4))
         + 0.07 * np.sin(TAU * 2.76 * f * t) * np.exp(-t / (dec * 0.25)))
    return amp * env * s


def pad_tone(f, t, detune=0.0):
    return (np.sin(TAU * f * t) + 0.35 * np.sin(TAU * 2 * f * t + 0.3) + 0.12 * np.sin(TAU * 3 * f * t + 1.1))


# ------------------------------------------------------------------ build
def main():
    cues = {"pings": [], "one_shots": []}
    ev = json.load(open(os.path.join(HERE, "data", "events_sim.json")))["bottom"]
    fx = np.zeros((2, N))          # effects (reverb send)
    dry = np.zeros((2, N))         # effects without reverb (swoosh)

    # ---------------- pendulum pings at every bottom pass, mapped through the clocks
    def ping_at(tv, i, gain, clock, tsim):
        p = -0.55 + 1.1 * i / 8
        add(fx, tv, pan(ping(NOTES[i], 0.22 * gain), p))
        cues["pings"].append(dict(tv=round(tv, 5), pendulum=i, fraction=FRACS[i], note_hz=NOTES[i], clock=clock,
                                  sim_t=round(tsim, 6), gain=round(gain, 3)))

    for key, ts in ev.items():
        i = int(key)
        for ts_ in ts:
            # live clock, main motion
            tv = ts_ + TL.T_PUSH
            if tv < 27.8:
                g = 0.3 + 0.7 * TL.pend_alpha(i, tv)
                if 18.0 <= tv:
                    g *= 0.7
                ping_at(tv, i, g, "live", ts_)
            # replay clock (0.5x): replayed pings, threshold pendulum loud, the dimmed others soft
            tv = TL.REPLAY_A + ts_ / TL.REPLAY_RATE
            if TL.REPLAY_A <= tv < TL.REPLAY_B:
                ping_at(tv, i, 0.9 if i == THR else 0.22, "replay", ts_)
            # live clock again for the final revelation
            tv = ts_ + TL.T_PUSH
            if TL.REPLAY_B <= tv < TL.REW_A:
                ping_at(tv, i, 0.3, "live", ts_)

    # ---------------- bloom as the phase portrait appears (17.6 s)
    t0 = 17.55
    d = 3.2
    t = np.arange(int(d * SR)) / SR
    env = np.clip(t / 0.7, 0, 1) ** 2 * np.exp(-np.maximum(t - 0.7, 0) / 0.9)
    bloom = sum(pad_tone(f, t) for f in (130.81, 196.0, 293.66, 329.63)) * env * 0.05
    nz = rng.standard_normal(len(t))
    nz = signal.sosfilt(signal.butter(2, [300, 2400], "bandpass", fs=SR, output="sos"), nz) * env * 0.05
    add(fx, t0, pan(bloom + nz, 0.0) * 1.0)
    cues["one_shots"].append(dict(name="bloom", tv=t0, dur=d))

    # ---------------- shimmer while the boundary traces itself (23.0 -> 25.0 s), panned left -> right with the trace
    sh_n = 0
    pent_hi = [1046.5, 1174.7, 1318.5, 1568.0, 1760.0, 2093.0, 2349.3]
    tt = 23.0
    while tt < 25.25:
        prog = TL.clamp((tt - 23.0) / 2.0)
        f = pent_hi[rng.integers(len(pent_hi))]
        g = np.arange(int(0.5 * SR)) / SR
        grain = np.sin(TAU * f * g) * (1 - np.exp(-g / 0.003)) * np.exp(-g / 0.12) * 0.045
        add(fx, tt, pan(grain, -0.8 + 1.6 * prog))
        sh_n += 1
        tt += 0.035 + 0.03 * rng.random()
    # soft glow swell when the boundary becomes the brightest thing (25.0 s)
    t = np.arange(int(2.6 * SR)) / SR
    env = np.clip(t / 0.4, 0, 1) * np.exp(-t / 0.9)
    add(fx, 24.95, pan(sum(np.sin(TAU * f * t) for f in (523.25, 783.99, 1046.5)) * env * 0.035, 0.0))
    cues["one_shots"].append(dict(name="shimmer", tv=23.0, dur=2.25, grains=sh_n))

    # ---------------- rising tone during the replay: pitch follows the gap to the top, then holds (never resolves)
    i0, i1 = int(TL.REPLAY_A * SR), int((TL.REW_A + 0.6) * SR)
    tv = np.arange(i0, i1) / SR
    tau = np.where(tv < TL.REPLAY_B, (tv - TL.REPLAY_A) * TL.REPLAY_RATE, tv - TL.T_PUSH)
    gap = 4 * np.arctan(np.exp(-W0 * tau))                     # computed directly
    semis = -19.0 * (gap / np.pi) ** 0.35
    f = 493.88 * 2 ** (semis / 12)                             # climbs to B4 (leading tone over the C drone) and holds
    ph = np.cumsum(TAU * f / SR)
    vib = 1 + 0.003 * np.sin(TAU * 5.0 * (tv - TL.REPLAY_A)) * np.clip((tv - 31) / 3, 0, 1)
    tone = np.sin(ph * vib) + 0.25 * np.sin(2 * ph + 0.4) + 0.08 * np.sin(3 * ph)
    amp = (0.17 * np.clip((tv - TL.REPLAY_A) / 0.6, 0, 1)
           * 10 ** (-2.5 * np.clip((tv - 33.0) / 7.0, 0, 1) / 20)          # very slightly quieter over time
           * np.clip((TL.REW_A + 0.45 - tv) / 0.45, 0, 1))                  # gives way to the rewind swoosh
    add(fx, TL.REPLAY_A, pan(tone * amp, 0.1))
    cues["one_shots"].append(dict(name="rising_tone", tv=TL.REPLAY_A, dur=TL.REW_A + 0.6 - TL.REPLAY_A,
                                  target_hz=493.88))

    # ---------------- reverse swoosh during the rewind (40.0 -> 41.7 s), ends in near-silence
    d = TL.REW_B - TL.REW_A
    t = np.arange(int(d * SR)) / SR
    u = t / d
    env = (np.exp(3.2 * u) - 1) / (np.exp(3.2) - 1)
    env *= np.clip((d - t) / 0.06, 0, 1) ** 2                              # short release at the end
    nz = rng.standard_normal((2, len(t)))
    # band sweeping upward (a reversed decay), soft highs
    out = np.zeros_like(nz)
    seg = 2400
    for k in range(0, len(t), seg):
        c = 300 + 2600 * (k / len(t)) ** 1.5
        sos = signal.butter(2, [c * 0.6, min(c * 1.6, 7000)], "bandpass", fs=SR, output="sos")
        out[:, k:k + seg] = signal.sosfilt(sos, nz[:, k:k + seg + 0], axis=-1)
    sw = out * env * 0.42
    # reversed ping chord riding inside the swoosh
    for i in range(9):
        p = ping(NOTES[i], 0.05, dur=d)[::-1]
        sw += pan(p * np.clip((d - t) / 0.06, 0, 1) ** 2, -0.55 + 1.1 * i / 8)
    add(dry, TL.REW_A, sw)
    cues["one_shots"].append(dict(name="reverse_swoosh", tv=TL.REW_A, dur=d))

    # ---------------- reverb on the effects (circular, so the loop stays seamless)
    ir_t = np.arange(int(2.2 * SR)) / SR
    ir = rng.standard_normal((2, len(ir_t))) * np.exp(-ir_t / 0.45)
    ir = np.stack([signal.sosfilt(signal.butter(2, 5000, "lowpass", fs=SR, output="sos"), r) for r in ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=-1, keepdims=True))
    wet = np.stack([circ_conv(fx[c], ir[c]) for c in range(2)])
    effects = fx + 0.35 * wet + dry

    # ---------------- ambient drone bed (exactly periodic over 42 s)
    t = np.arange(N) / SR
    drone = np.zeros((2, N))
    for f, a, p in ((65.41, 1.0, -0.2), (98.0, 0.6, 0.2), (130.81, 0.45, -0.1), (196.0, 0.18, 0.3), (164.81, 0.10, -0.3)):
        for det, side in ((0.0, 0), (3.0 / DUR, 1)):
            fq = q42(f) + det
            lfo = 1 + 0.25 * np.sin(TAU * t * (2 / DUR) + f)
            drone[side] += a * lfo * np.sin(TAU * fq * t + f)
    drone *= 0.5
    # bed envelope: near-silence at the loop seam, gentle swell under the map, dip in the replay
    def benv(x):
        e = np.full_like(x, 1.0)
        e *= 0.06 + 0.94 * np.clip((x - 0.3) / 1.7, 0, 1) ** 2
        e *= 0.06 + 0.94 * np.clip((TL.REW_B - 0.05 - x) / 1.6, 0, 1) ** 2
        e *= 1 + 0.35 * np.clip((x - 17.5) / 1.5, 0, 1) * np.clip((28.0 - x) / 1.0, 0, 1)
        e *= 1 - 0.35 * np.clip((x - 28.0) / 1.0, 0, 1) * np.clip((TL.REW_A - x) / 0.5, 0, 1)
        return e
    bed = drone * benv(t)

    # ---------------- mix: bed ~9 LU under the effects
    meter = pyln.Meter(SR)
    l_fx = meter.integrated_loudness(effects.T)
    l_bed = meter.integrated_loudness(bed.T)
    bed *= 10 ** ((l_fx - 9.0 - l_bed) / 20)
    mix = effects + bed
    # phone-friendly: high-pass below 40 Hz, gentle high shelf down above 9 kHz (circular -> no seam)
    mix = circ_filter(signal.butter(4, 40, "highpass", fs=SR, output="sos"), mix)
    mix = circ_filter(signal.butter(2, 9000, "lowpass", fs=SR, output="sos"), mix)

    # ---------------- loudness to -14 LUFS, true peak <= -1 dBTP (4x oversampled look-ahead limiter)
    def true_peak(x):
        up = signal.resample_poly(x, 4, 1, axis=-1)
        return 20 * np.log10(np.max(np.abs(up)) + 1e-12)

    def limit(x, ceiling_db=-1.6):
        c = 10 ** (ceiling_db / 20)
        up = np.max(np.abs(signal.resample_poly(x, 4, 1, axis=-1)), axis=0)
        pk = up.reshape(-1, 4).max(axis=1)[:N]
        g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
        la = int(0.004 * SR)
        # look-ahead min filter then smooth release
        from scipy.ndimage import minimum_filter1d
        g = minimum_filter1d(g, size=2 * la + 1, mode="wrap")
        rel = np.exp(-1 / (0.08 * SR))
        out = np.empty_like(g)
        prev = 1.0
        for i in range(len(g)):
            prev = g[i] if g[i] < prev else g[i] + (prev - g[i]) * rel
            out[i] = prev
        out = np.convolve(out, np.ones(la) / la, mode="same")
        return x * out

    for _ in range(4):
        L = meter.integrated_loudness(mix.T)
        mix *= 10 ** ((-14.0 - L) / 20)
        if true_peak(mix) > -1.6:
            mix = limit(mix)
    L = meter.integrated_loudness(mix.T)
    tp = true_peak(mix)
    seam = float(np.max(np.abs(mix[:, 0] - mix[:, -1])))
    edge = dict(first_ms_rms=float(np.sqrt(np.mean(mix[:, :480] ** 2))), last_ms_rms=float(np.sqrt(np.mean(mix[:, -480:] ** 2))))

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    pcm = np.clip(mix.T, -1, 1)
    import wave
    with wave.open(os.path.join(HERE, "out", "mix.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        b = (pcm * (2 ** 23 - 1)).astype(np.int32)
        w.writeframes(np.stack([(b >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.uint8).tobytes())
    cues["mix"] = dict(samples=N, seconds=N / SR, lufs_pyloudnorm=L, true_peak_dbtp_4x=tp,
                       bed_under_effects_lu=9.0, loop_seam_sample_jump=seam, **edge)
    json.dump(cues, open(os.path.join(HERE, "data", "audio_cues.json"), "w"), indent=1)
    print(f"pings: {len(cues['pings'])}  LUFS={L:.2f}  TP={tp:.2f} dBTP  seam jump={seam:.2e}  edges={edge}")


if __name__ == "__main__":
    main()
