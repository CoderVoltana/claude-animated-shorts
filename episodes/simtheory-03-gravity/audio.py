"""SimTheory #03 - synthesized sound, timed from the same physics as the picture.

- drone: additive partials on the 1/DUR Hz grid (exactly periodic over the loop) + periodic noise air
- ticks: one per full swing from the RK4 return times (low = Earth, high = 10x), panned to their side
- sync: both ticks land together on one A-major chord, then a brief silence
- bloom when the formula appears; buzzing whir for the 100x ghost driven by its simulated angular speed
- reverb is a circular convolution over the whole loop, so tails wrap around seamlessly
- master: BS.1770 integrated loudness -> -14 LUFS, 4x-oversampled true-peak limiter <= -1 dBTP
Writes build/audio.wav (48 kHz, stereo, 24-bit, exactly DUR seconds) and prints a loudness report.
"""
import math
import os
import wave

import numpy as np
from scipy import signal

import sim
import timeline as TL

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
N = int(round(TL.DUR * SR))
t = np.arange(N) / SR
rng = np.random.default_rng(7)


def at(sec):
    return int(round(sec * SR))


def add(buf, x, start, pan=0.0, gain=1.0):
    """mix mono x into stereo buf at sample start (constant-power pan, wraps around the loop)"""
    l = math.cos((pan + 1) * math.pi / 4) * gain
    r = math.sin((pan + 1) * math.pi / 4) * gain
    idx = (start + np.arange(len(x))) % N
    np.add.at(buf[0], idx, x * l)
    np.add.at(buf[1], idx, x * r)


def env_keys(keys):
    ks = np.array(keys)
    return np.interp(t, ks[:, 0], ks[:, 1])


# ---------------------------------------------------------------------------------------------
def drone():
    out = np.zeros((2, N))
    base = 1.0 / TL.DUR                       # frequency grid -> every partial loops exactly
    parts = [(55.0, 0.60), (55.1, 0.18), (82.5, 0.35), (110.0, 0.50), (110.2333, 0.18), (165.0, 0.25),
             (220.0, 0.12), (247.5, 0.05), (330.0, 0.05)]
    for ch in range(2):
        for f, a in parts:
            f = round(f / base) * base
            if ch == 1:
                f += base * (1 if f > 100 else 0)  # tiny L/R difference for width
            ph = rng.uniform(0, 2 * math.pi)
            out[ch] += a * np.sin(2 * math.pi * f * t + ph)
    # slow periodic movement (periods divide DUR)
    lfo = 0.80 + 0.12 * np.sin(2 * math.pi * t / 7.5) + 0.08 * np.sin(2 * math.pi * t / 10.0 + 1.0)
    out *= lfo
    # periodic band-limited noise "air" built on the loop's FFT grid
    for ch in range(2):
        spec = np.zeros(N // 2 + 1, complex)
        f = np.fft.rfftfreq(N, 1 / SR)
        band = np.exp(-0.5 * (np.log(np.maximum(f, 1) / 420.0) / 0.55) ** 2)
        spec[:] = band * np.exp(1j * rng.uniform(0, 2 * math.pi, len(f)))
        n = np.fft.irfft(spec, N)
        out[ch] += 0.10 * n / np.std(n) * 0.35
    e = env_keys([(0, 0.07), (1.7, 1.0), (TL.T_SYNC - 0.02, 1.0), (TL.T_SYNC + 0.06, 0.0),
                  (TL.T_FORMULA - 0.05, 0.0), (TL.T_FORMULA + 1.6, 0.85), (21.6, 0.85), (22.0, 0.70),
                  (26.9, 0.70), (TL.DUR, 0.07)])
    return out * e * 0.11


def woodtick(f0, dur=0.16, decay=0.045):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    pitch = f0 * (1 + 0.06 * np.exp(-tt / 0.004))
    ph = 2 * math.pi * np.cumsum(pitch) / SR
    body = np.sin(ph) * np.exp(-tt / decay) + 0.35 * np.sin(2.76 * ph) * np.exp(-tt / (decay * 0.5))
    click = rng.standard_normal(n) * np.exp(-tt / 0.0015)
    b, a = signal.butter(2, [min(f0 * 3, 9000) / (SR / 2), min(f0 * 9, 20000) / (SR / 2)], "band")
    click = signal.lfilter(b, a, click)
    x = body + 0.5 * click
    x[: int(0.0015 * SR)] *= np.linspace(0, 1, int(0.0015 * SR))
    return x / np.max(np.abs(x))


def chord():
    dur = 1.15
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n)
    for f, a, tau in [(220.0, 1.0, 0.55), (277.18, 0.7, 0.45), (329.63, 0.75, 0.45), (440.0, 0.55, 0.35),
                      (659.26, 0.30, 0.25), (880.0, 0.16, 0.18)]:
        x += a * (np.sin(2 * math.pi * f * tt) + 0.15 * np.sin(2 * math.pi * 2 * f * tt)) * np.exp(-tt / tau)
    att = np.minimum(tt / 0.004, 1)
    rel = np.clip((dur - tt) / 0.35, 0, 1) ** 2
    return x * att * rel / np.max(np.abs(x))


def bloom():
    dur = 4.2
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = np.zeros((2, n))
    for ch in range(2):
        for f, a in [(110, 0.7), (164.81, 0.55), (220, 0.6), (277.18, 0.35), (329.63, 0.4), (493.88, 0.22),
                     (880, 0.08), (1318.5, 0.05)]:
            det = 1 + (0.0025 if ch else -0.0025)
            x[ch] += a * np.sin(2 * math.pi * f * det * tt + rng.uniform(0, 6.28))
            x[ch] += 0.5 * a * np.sin(2 * math.pi * f / det * tt + rng.uniform(0, 6.28))
    swell = (1 - np.exp(-tt / 0.28)) * np.exp(-np.maximum(tt - 0.55, 0) / 1.3)
    shimmer = 1 + 0.15 * np.sin(2 * math.pi * 5.5 * tt)
    x *= swell * shimmer
    # airy noise whoosh rising into the bloom
    nz = rng.standard_normal(n)
    b, a = signal.butter(2, [1500 / (SR / 2), 7000 / (SR / 2)], "band")
    nz = signal.lfilter(b, a, nz) * np.exp(-((tt - 0.35) / 0.25) ** 2) * 0.25
    x += nz
    x[:, -int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    return x / np.max(np.abs(x))


def whir():
    """buzz for the 100x ghost, its loudness and pitch follow the simulated |omega| of the ghost"""
    out = np.zeros((2, N))
    a0, a1 = at(TL.GHOST_IN[0]), at(TL.GHOST_OUT[1])
    tt = t[a0:a1]
    tau = tt - TL.GHOST_REL
    tr = sim.Traj(sim.G_GHOST, 7.0)
    i = np.clip((tau / tr.dt).astype(int), 0, len(tr.om) - 1)
    om = np.abs(tr.om[i]) / np.max(np.abs(tr.om))
    th = tr.th[i] / sim.THETA0
    f = 92 + 46 * om
    ph = 2 * math.pi * np.cumsum(f) / SR
    saw = sum(np.sin(k * ph) / k for k in range(1, 18))
    b, a = signal.butter(2, [120 / (SR / 2), 2400 / (SR / 2)], "band")
    buzz = signal.lfilter(b, a, saw)
    nz = signal.lfilter(*signal.butter(2, [700 / (SR / 2), 5000 / (SR / 2)], "band"), rng.standard_normal(len(tt)))
    x = buzz / np.std(buzz) * (0.35 + 0.65 * om) + 0.5 * nz / np.std(nz) * om ** 2
    vis = np.array([TL.ghost_alpha(s) for s in tt])
    vis = np.convolve(vis, np.ones(96) / 96, "same")       # 2 ms smoothing, no clicks on the flicker
    x *= vis
    pan = 0.35 * th                                       # follows the bob left/right
    out[0, a0:a1] = x * np.cos((pan + 1) * math.pi / 4)
    out[1, a0:a1] = x * np.sin((pan + 1) * math.pi / 4)
    return out


def blip(f0, f1, dur=0.14):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (tt / dur)
    x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.05) * np.minimum(tt / 0.003, 1)
    return x


def circ_reverb(x, rt=1.3, seed=3):
    """stereo reverb as a circular convolution over the loop (tails wrap -> seamless loop)"""
    n_ir = int(rt * 1.5 * SR)
    tt = np.arange(n_ir) / SR
    r = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for ch in range(2):
        ir = r.standard_normal(n_ir) * np.exp(-6.9 * tt / rt)
        ir[: int(0.012 * SR)] = 0                        # pre-delay
        ir = signal.lfilter(*signal.butter(1, 5000 / (SR / 2)), ir)
        ir /= np.sqrt(np.sum(ir ** 2))
        irp = np.zeros(N)
        irp[:n_ir] = ir
        out[ch] = np.fft.irfft(np.fft.rfft(x[ch]) * np.fft.rfft(irp), N)
    return out


# ---- loudness (ITU-R BS.1770-4) --------------------------------------------------------------
def k_weight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def integrated_lufs(x):
    y = k_weight(x)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = []
    for s in range(0, y.shape[1] - blk + 1, hop):
        ms.append(np.sum(np.mean(y[:, s:s + blk] ** 2, axis=1)))
    ms = np.array(ms)
    lk = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(np.mean(g2))


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=-1)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-20)


def limiter(x, ceil_db=-1.3, look=0.003, rel=0.08):
    """look-ahead peak limiter driven by the 4x-oversampled peak envelope (circular, loop-safe)"""
    ceil = 10 ** (ceil_db / 20)
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(axis=0)
    pk = up.reshape(-1, 4).max(axis=1)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    w = int(look * SR)
    # hold the minimum over the look-ahead window (centre-aligned, circular)
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    g = minimum_filter1d(need, size=2 * w + 1, mode="wrap")
    g = uniform_filter1d(g, size=w, mode="wrap")
    # release smoothing
    a = math.exp(-1 / (rel * SR))
    gs = np.empty_like(g)
    prev = 1.0
    for _ in range(2):                                   # two passes -> state wraps around the loop
        for i in range(len(g)):
            prev = g[i] if g[i] < prev else a * prev + (1 - a) * g[i]
            gs[i] = prev
    return x * gs, float(gs.min())


# ---------------------------------------------------------------------------------------------
def build():
    dry = np.zeros((2, N))
    send = np.zeros((2, N))
    events = []

    tick_lo = woodtick(220.0, decay=0.05)
    tick_hi = woodtick(659.26, decay=0.035)
    for te in TL.CNT_E[:-1]:
        add(dry, tick_lo, at(te), -0.35, 0.55); add(send, tick_lo, at(te), -0.35, 0.25)
        events.append(("tick EARTH", te))
    for te in TL.CNT_H[:-1]:
        add(dry, tick_hi, at(te), 0.35, 0.36); add(send, tick_hi, at(te), 0.35, 0.18)
        events.append(("tick 10x", te))
    # sync: both ticks together + chord
    ts = TL.T_SYNC
    add(dry, tick_lo, at(ts), -0.35, 0.55); add(dry, tick_hi, at(ts), 0.35, 0.36)
    c = chord()
    add(dry, c, at(ts), -0.15, 0.42); add(dry, c, at(ts) + 7, 0.15, 0.42)
    add(send, c, at(ts), 0.0, 0.18)
    events.append(("sync chord", ts))

    # soft release thump
    th = blip(140, 70, 0.18)
    add(dry, th, at(TL.T_REL), 0, 0.35)
    events.append(("release", TL.T_REL))

    b = bloom()
    s0 = at(TL.T_FORMULA - 0.05)
    for ch in range(2):
        idx = (s0 + np.arange(b.shape[1])) % N
        dry[ch, idx] += 0.40 * b[ch]
        send[ch, idx] += 0.30 * b[ch]
    events.append(("formula bloom", TL.T_FORMULA))

    w = whir()
    dry += 0.16 * w
    send += 0.08 * w
    events.append(("ghost whir", TL.GHOST_IN[0]))

    bl = blip(880, 440, 0.16)
    add(dry, bl, at(TL.T_RESET), -0.3, 0.16); add(dry, bl, at(TL.T_RESET) + 240, 0.3, 0.16)
    add(send, bl, at(TL.T_RESET), 0, 0.12)
    events.append(("counter reset", TL.T_RESET))

    d = drone()
    mix = d + dry + circ_reverb(send, rt=1.2)

    # master
    for _ in range(4):
        lu = integrated_lufs(mix)
        mix *= 10 ** ((-14.0 - lu) / 20)
        mix, gmin = limiter(mix, ceil_db=-2.1)   # kit EXPORT true_peak -1.5 dBTP, with headroom for AAC
    lu = integrated_lufs(mix)
    tp = true_peak_db(mix)
    return mix, lu, tp, gmin, events


def write_wav24(path, x):
    y = np.clip(x, -1, 1 - 2 ** -23)
    q = np.round(y.T * (2 ** 23 - 1)).astype(np.int32)
    b = np.ascontiguousarray(q.astype("<i4")).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(b)


if __name__ == "__main__":
    mix, lu, tp, gmin, events = build()
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    p = os.path.join(HERE, "build", "audio.wav")
    write_wav24(p, mix)
    seam = np.abs(mix[:, 0] - mix[:, -1]).max()
    typ = np.abs(np.diff(mix[:, -SR:], axis=1)).max()
    edge_db = 20 * np.log10(np.sqrt(np.mean(mix[:, -SR // 2:] ** 2)) + 1e-12)
    print(f"wrote {p}: {mix.shape[1] / SR:.3f} s")
    print(f"integrated {lu:.2f} LUFS, true peak {tp:.2f} dBTP, limiter max GR {20 * math.log10(gmin):.2f} dB")
    print(f"loop seam |x[0]-x[-1]| = {seam:.5f} (max step in last second {typ:.5f}); last 0.5 s RMS {edge_db:.1f} dBFS")
    for n, s in events:
        print(f"  {s:7.3f} s  {n}")
