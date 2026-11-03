"""SimTheory #02 soundtrack, fully synthesized (numpy/scipy), timed from timeline.py and the measured simulation.

Layers: periodic drone (every partial is a whole number of cycles in 30 s -> sample-continuous loop), forward-run
riser + swing whooshes driven by the real bob speed + accelerating clock ticks, tape-stop at the freeze, reversed
"rewind" swoosh ending on the velocity flip followed by silence, a calm steady chord while the copies are aligned,
detuned split voices entering exactly on each measured 2-degree crossing (detune follows the measured separation),
a deep bloom hit on the graph, and a fade to near-silence as the ghost lands.

Master: -14 LUFS integrated (BS.1770-4), true peak <= -1 dBTP (4x oversampled look-ahead limiter). 48 kHz, 24-bit.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage, signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "templates", "audio"))   # dsp.py (append: its timeline.py must not shadow ours)
from dsp import (SR, ns, dbg, rng, lp, hp, bp, colored, st, pan, make_ir, integrated, true_peak, write_wav,  # noqa: E402
                 short_term, momentary)

import timeline as TL                                                                                          # noqa: E402
from config import DUR, T_SIM                                                                                  # noqa: E402
from physics import DT, all_omegas                                                                             # noqa: E402

N = ns(DUR)
t = np.arange(N) / SR
TARGET = -14.0
CEIL = -1.3
D = np.load(os.path.join(HERE, "sim.npz"))
EV = TL.EVENTS


def sstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * u * (u * (6 * u - 15) + 10)


def osc(freq, phase0=0.0):
    """sine with time-varying frequency (array or scalar) via phase accumulation"""
    f = np.broadcast_to(np.asarray(freq, float), (N,))
    ph = phase0 + 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph)


def saw(freq, nh=24):
    f = np.broadcast_to(np.asarray(freq, float), (N,))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(N)
    for k in range(1, nh + 1):
        out += np.sin(k * ph) / k * (k * f < SR * 0.45)
    return out * 0.55


def tv_lowpass(x, fc, block=256):
    """time-varying one-pole-pair lowpass (cutoff array), blockwise biquad with state carried over"""
    y = np.zeros_like(x)
    zi = np.zeros((1, 2))
    for i in range(0, len(x), block):
        f = float(np.clip(fc[i], 30, SR * 0.45))
        sos = signal.butter(2, f, "lowpass", fs=SR, output="sos")
        y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return y


def circ_reverb(x, rt, seed, wet=1.0):
    """reverb by CIRCULAR convolution over the 30 s loop, so any tail wraps into the start seamlessly"""
    ir = make_ir(rt, seed, length=min(rt * 1.3, DUR - 1))
    X = st(x)
    out = np.zeros_like(X)
    for c in range(2):
        h = np.zeros(N)
        h[: ir.shape[1]] = ir[c]
        out[c] = np.fft.irfft(np.fft.rfft(X[c]) * np.fft.rfft(h), N)
    return out * wet


# ------------------------------------------------------------------ 1. drone (loop-periodic)
def drone():
    parts = [(55.0, 1.0), (82.5, 0.35), (110.0, 0.55), (165.0, 0.18), (220.0, 0.10)]
    L = np.zeros(N); R = np.zeros(N)
    for f, a in parts:
        L += a * osc(f)
        R += a * osc(f + (1.0 / 30.0) * (2 if f > 60 else 1), 0.7)   # slow 30 s / 15 s beating, still periodic
    lfo = 0.82 + 0.18 * np.cos(2 * np.pi * t / 10.0)                 # 3 cycles per loop
    x = np.stack([L, R]) * lfo
    x = np.tanh(1.3 * x) / 1.3
    L0 = 0.30                                                        # loop-point ("near-silence") level
    env = (L0 + (0.75 - L0) * sstep(0.0, 2.0, t)
           + 0.25 * sstep(2.0, 8.8, t)                               # rising energy
           - 0.40 * sstep(8.98, 9.05, t)                             # freeze
           - 0.58 * sstep(9.84, 9.9, t) + 0.58 * sstep(10.25, 10.9, t)   # the silence right after the flip
           - (0.60 - L0) * sstep(27.0, 29.4, t))
    env = np.maximum(env, 0.0)
    return x * env * 0.22, env


# ------------------------------------------------------------------ 2. forward run: riser, whooshes, ticks
def lower_bob_speed_forward():
    """|v2| of the real pendulum at each audio sample of the forward run (playback speed included)."""
    ctrl = np.arange(0, DUR, 0.002)
    sims = np.array([TL.sim_forward(x) for x in ctrl])
    idx = np.clip(np.round(sims / DT).astype(int), 0, len(D["F"]) - 1)
    Y = D["F"][idx]
    W = all_omegas(Y)
    vx = np.cos(Y[:, 0]) * W[:, 0] + np.cos(Y[:, 1]) * W[:, 1]
    vy = np.sin(Y[:, 0]) * W[:, 0] + np.sin(Y[:, 1]) * W[:, 1]
    v = np.hypot(vx, vy) * np.array([TL.speed(x) for x in ctrl])   # apparent speed on screen
    return np.interp(t, ctrl, v)


def forward_run():
    r = rng(7)
    on = sstep(2.0, 2.2, t) * (1 - sstep(9.0, 9.22, t))
    prog = np.clip((t - 2.0) / 7.0, 0, 1)
    # tape-stop at the freeze: pitch falls over 0.22 s
    stop = np.where(t < 9.0, 1.0, np.maximum(0.2, 1 - (t - 9.0) / 0.22 * 0.8))
    base = 110.0 * 2 ** (prog * 3 / 12) * stop
    chord = saw(base) + 0.7 * saw(base * 1.5 * 1.003) + 0.5 * saw(base * 2 * 0.997)
    fc = 220 * (16 ** prog) * stop
    riser = tv_lowpass(chord, fc) * (0.10 + 0.9 * prog ** 1.6) * on
    # whooshes = band noise following the real lower-bob speed
    v = lower_bob_speed_forward()
    vn = np.clip(v / np.percentile(v[(t > 2.5) & (t < 9)], 95), 0, 1.3)
    noise = colored(N, r, 1.0)
    wl = bp(noise, 350, 3000, 2) * vn ** 2 * on * (0.5 + 0.5 * prog)
    wr = bp(colored(N, r, 1.0), 350, 3000, 2) * vn ** 2 * on * (0.5 + 0.5 * prog)
    # clock ticks accelerating with the run
    rate = 3.0 + 7.0 * prog ** 1.3
    ph = np.cumsum(rate) / SR
    ticks = np.zeros(N)
    kk = np.flatnonzero(np.diff(np.floor(ph)) > 0)
    burst = np.sin(2 * np.pi * 2400 * np.arange(ns(0.012)) / SR) * np.exp(-np.arange(ns(0.012)) / ns(0.003))
    for k in kk:
        if 2.0 <= t[k] < 9.0:
            seg = burst[: N - k]
            ticks[k:k + len(seg)] += seg * (0.4 + 0.6 * prog[k])
    out = st(riser) * 0.30 + np.stack([wl, wr]) * 0.20 + pan(ticks, 0.25) * 0.10
    # release pluck at 2.0 s
    pk = np.zeros(N)
    n0 = ns(2.0)
    tt = np.arange(ns(1.2)) / SR
    pk[n0:n0 + len(tt)] = (np.sin(2 * np.pi * 440 * tt) + 0.5 * np.sin(2 * np.pi * 880 * tt)) * np.exp(-tt / 0.25)
    out += st(pk) * 0.12
    return out


# ------------------------------------------------------------------ 3. rewind swoosh + snap + silence
def swoosh():
    r = rng(11)
    flip = TL.T_FLIP
    d = 0.55
    n = ns(d)
    tt = np.arange(n) / SR
    z = colored(n, r, 0.5)
    # forward version: bright burst decaying with a falling band, then time-reversed -> swell into the flip
    fwd = z * np.exp(-tt / 0.11)
    fc = 9000 * np.exp(-tt / 0.18) + 400
    fwd = tv_lowpass_local(fwd, fc)
    fwd = hp(fwd, 250, 2)
    rev = fwd[::-1]
    rev /= np.max(np.abs(rev)) + 1e-12
    out = np.zeros((2, N))
    i1 = ns(flip)
    i0 = i1 - n
    L = rev * np.linspace(1.0, 0.6, n)
    R = rev * np.linspace(0.6, 1.0, n)                               # sweeps right -> left (a "rewind")
    out[0, i0:i1] += L
    out[1, i0:i1] += R
    # snap: click + thump exactly on the flip
    m = ns(0.18)
    ts = np.arange(m) / SR
    snap = 0.9 * np.exp(-ts / 0.004) * colored(m, r, 0.0) + np.sin(2 * np.pi * (140 - 300 * ts) * ts) * np.exp(-ts / 0.05)
    out[:, i1:i1 + m] += st(snap) * 0.6
    return out * 0.40


def tv_lowpass_local(x, fc, block=128):
    y = np.zeros_like(x)
    zi = np.zeros((1, 2))
    for i in range(0, len(x), block):
        sos = signal.butter(2, float(np.clip(fc[i], 30, SR * 0.45)), "lowpass", fs=SR, output="sos")
        y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return y


# ------------------------------------------------------------------ 4. calm tone + detuned splits
CHORD = [(220.0, 1.0), (329.63, 0.55), (440.0, 0.35), (554.37, 0.12)]


def pad(freq_mult, phase_seed=0.0):
    x = np.zeros(N)
    for i, (f, a) in enumerate(CHORD):
        x += a * osc(f * freq_mult, phase_seed + i)
        x += 0.18 * a * osc(2 * f * freq_mult, phase_seed + 2 * i)
    return x / 2.2


def separation_cents(ang, cross_v, sign):
    """detune follows the MEASURED separation of that copy (log scale) from the moment it crosses 2 deg."""
    ctrl = np.arange(0, DUR, 0.005)
    tau = np.array([TL.tau_rev(x) for x in ctrl])
    k = np.clip(np.round(tau / DT).astype(int), 0, len(ang) - 1)
    sep = ndimage.maximum_filter1d(ang[k], 40)
    lg = np.clip(np.log10(np.maximum(sep, 1e-20) / math.radians(2.0)) / 1.5, 0, 1)   # 2 deg -> 0, ~60 deg -> 1
    cents = sign * (22 + 48 * lg)
    c = np.interp(t, ctrl, cents)
    c = ndimage.uniform_filter1d(c, ns(0.08))
    return c


def tones():
    ghost_env = sstep(11.0, 11.9, t) * (1 - sstep(27.2, 29.3, t))
    calm = pad(1.0) * ghost_env
    vo, vc = EV["orange_cross"], EV["cyan_cross"]
    co = separation_cents(D["ang_o"], vo, +1)
    cc = separation_cents(D["ang_c"], vc, -1)
    eo = sstep(vo, vo + 0.03, t) * (1 - sstep(27.0, 28.0, t))
    ec = sstep(vc, vc + 0.03, t) * (1 - sstep(27.0, 28.0, t))
    orange = pad(2 ** (co / 1200), 0.3) * eo
    cyan = pad(2 ** (cc / 1200), 0.9) * ec
    # an audible "crack" articulation on each split: short bright partial
    for v0 in (vo, vc):
        n0 = ns(v0)
        tt = np.arange(ns(0.6)) / SR
        cr = np.sin(2 * np.pi * 1318.5 * tt) * np.exp(-tt / 0.09) * 0.5
        orange[n0:n0 + len(tt)] += cr if v0 == vo else 0
        cyan[n0:n0 + len(tt)] += cr if v0 == vc else 0
    # the steady tone gives up some level as the copies leave it (keeps the sum even)
    calm *= 1 - 0.25 * eo - 0.2 * ec
    out = st(calm) * 0.30 + pan(orange, 0.55) * 0.27 + pan(cyan, -0.55) * 0.27
    return out


# ------------------------------------------------------------------ 5. bloom hit
def bloom():
    r = rng(23)
    t0 = TL.T_GRAPH
    n = ns(4.0)
    tt = np.arange(n) / SR
    f = 38 + 30 * np.exp(-tt / 0.25)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 1.4) * (1 - np.exp(-tt / 0.004))
    whump = lp(colored(n, r, 1.5), 260, 2) * np.exp(-tt / 0.35) * 0.6
    shimmer = sum(a * np.sin(2 * np.pi * fr * tt) for fr, a in ((880, 0.5), (1318.5, 0.35), (1108.7, 0.3), (1760, 0.15)))
    shimmer *= (1 - np.exp(-tt / 0.06)) * np.exp(-tt / 1.1) * 0.25
    x = np.zeros(N)
    i0 = ns(t0)
    m = min(n, N - i0)
    x[i0:i0 + m] = (sub + whump)[:m]
    sh = np.zeros(N)
    sh[i0:i0 + m] = shimmer[:m]
    return st(x) * 0.55, st(sh)


# ------------------------------------------------------------------ master
def limiter(x, ceil_db=CEIL, look=0.003, rel=0.08):
    c = dbg(ceil_db)
    n = x.shape[1]
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(0)
    pk = np.pad(up, (0, max(0, 4 * n - len(up))))[: 4 * n].reshape(n, 4).max(1)
    gr = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    La = max(1, ns(look))
    g1 = ndimage.minimum_filter1d(gr, 2 * La + 1, mode="wrap")
    # release: one-pole recovery (vectorised via running minimum of a decaying ramp is overkill here)
    g = np.empty(n)
    kr = 1 - math.exp(-1 / (rel * SR))
    acc = 1.0
    for i in range(n):
        acc = min(g1[i], acc + (1 - acc) * kr)
        g[i] = acc
    g = ndimage.uniform_filter1d(g, 2 * La + 1, mode="wrap")
    return x * np.minimum(g, 1.0)[None, :], g


def main():
    dr, denv = drone()
    fw = forward_run()
    sw = swoosh()
    tn = tones()
    bl, sh = bloom()
    dry = dr + fw + sw + tn + bl + sh
    send = 0.25 * fw + 0.35 * sw + 0.55 * tn + 0.25 * bl + 1.0 * sh
    wet = circ_reverb(send, 2.6, 5) * 0.35
    mix = dry + wet
    mix = hp(np.concatenate([mix, mix, mix], axis=1), 24, 2)[:, N:2 * N]   # filtered circularly (loop-safe)
    # keep the loop seam exact: crossfade-free because every non-drone layer is ~0 at both ends;
    # remove any residual DC so the wrap is continuous
    mix -= mix.mean(axis=1, keepdims=True)
    G = 0.0
    for it in range(12):
        y, g = limiter(mix * dbg(G))
        L = integrated(y)
        if abs(L - TARGET) < 0.03:
            break
        G += TARGET - L
    tp = true_peak(y)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    write_wav(os.path.join(HERE, "build", "mix.wav"), y)
    seam = np.max(np.abs(y[:, 0] - y[:, -1]))
    typ = np.median(np.max(np.abs(np.diff(y, axis=1)), axis=0))
    rep = dict(integrated_lufs=float(L), true_peak_dbtp=float(tp), master_gain_db=float(G),
               max_gain_reduction_db=float(-20 * np.log10(g.min())), loop_seam_jump=float(seam),
               median_sample_step=float(typ), level_first_100ms=float(20 * np.log10(np.sqrt(np.mean(y[:, :4800] ** 2)) + 1e-12)),
               level_last_100ms=float(20 * np.log10(np.sqrt(np.mean(y[:, -4800:] ** 2)) + 1e-12)))
    json.dump(rep, open(os.path.join(HERE, "build", "audio_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
