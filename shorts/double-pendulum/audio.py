"""Original synthesized score + SFX for the double-pendulum short (48 kHz stereo, exactly 24.0 s).

Beds are rendered 2 s past the end and the overhang is crossfaded onto the start, so the
audio loops seamlessly with the picture. Event cues follow the picture's timeline (render.html).
Cue times below are STORY times; O shifts them into video time.

  python3 audio.py        -> score.wav       (default 24 s, ends with the rewind)
  python3 audio.py cold   -> score_cold.wav  (23.2 s cold open: 0.5 s chaos flash, 0.5 s rewind,
                                              story from 1.0 s, ends mid-chaos and wraps to the flash)
"""
import sys, numpy as np, wave

COLD = len(sys.argv) > 1 and sys.argv[1] == "cold"
SR = 48000
O = 1.0 if COLD else 0.0           # story t = video time - O
T = 23.2 if COLD else 24.0
EXT = 2.0
XF = 0.7 if COLD else 2.0          # bed loop crossfade length
END = 1e9 if COLD else 22.2        # reveal energy never ends in the cold open (it wraps to the flash)
n = int((T + EXT) * SR)
t = np.arange(n) / SR
st = t - O
rng = np.random.default_rng(7)

def env(x, a, b, fi, fo):
    """1 inside [a,b] with linear fades."""
    return np.clip((x - a) / fi, 0, 1) * np.clip((b - x) / fo, 0, 1)

def lp_fast(x, cutoff):
    """zero-phase 2nd-order low-pass via FFT"""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / cutoff) ** 4), len(x))

def hp_fast(x, cutoff):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X * (f / cutoff) ** 2 / np.sqrt(1 + (f / cutoff) ** 4), len(x))

def reverb(x, secs=2.5, decay=3.0, seed=1):
    r = np.random.default_rng(seed)
    m = int(secs * SR); k = np.arange(m) / SR
    out = []
    for ch in range(2):
        ir = r.standard_normal(m) * np.exp(-decay * k / secs * 2.3)
        ir = lp_fast(ir, 5000); ir /= np.sqrt((ir ** 2).sum())
        L = len(x) + m
        y = np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(ir, L), L)[:len(x)]
        out.append(y)
    return np.stack(out)

def note(f, start, dur, amp=1.0, kind="pluck", det=0.0):
    i0 = int((start + O) * SR); m = int(dur * SR)
    k = np.arange(m) / SR
    if kind == "pluck":
        e = np.exp(-k * 5.5) * np.clip(k / 0.004, 0, 1)
        s = np.sin(2 * np.pi * f * k) + 0.35 * np.sin(2 * np.pi * 2 * f * k) * np.exp(-k * 9) + 0.15 * np.sin(2 * np.pi * 3.01 * f * k) * np.exp(-k * 14)
    elif kind == "bell":
        e = np.exp(-k * 2.2) * np.clip(k / 0.003, 0, 1)
        s = np.sin(2 * np.pi * f * k) + 0.5 * np.sin(2 * np.pi * 2.76 * f * k) * np.exp(-k * 4) + 0.25 * np.sin(2 * np.pi * 5.4 * f * k) * np.exp(-k * 7)
    y = np.zeros(n); y[i0:i0 + m] = (s * e * amp)[: max(0, min(m, n - i0))]
    return y

def mtof(m): return 440 * 2 ** ((m - 69) / 12)

# ---------------- continuous beds (loop-crossfaded) ----------------
bed = np.zeros((2, n))
# pad: D minor(add9) voicing, slow chorus, swells with the story
pad_notes = [38, 45, 50, 53, 57, 64]
energy = (0.55 + 0.25 * env(st, 2, 16.8, 2, 0.4) + 0.6 * env(st, 17, END, 0.05, 1.2))
pad = np.zeros((2, n))
for j, m in enumerate(pad_notes):
    for ch, d in enumerate((-0.11, 0.13)):
        f = mtof(m) * (1 + d / 100 * (j + 1))
        ph = 2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * (0.13 + 0.05 * j) * t + j)
        saw = sum(np.sin(h * ph) / h for h in range(1, 7))
        pad[ch] += saw * (0.22 if m < 45 else 0.12)
for ch in range(2):
    pad[ch] = lp_fast(pad[ch], 900)
bright = lp_fast(np.stack([hp_fast(p, 1200) for p in pad]).sum(0), 6000)
bed += pad * energy * 0.32 + 0.10 * bright * env(st, 17, END, 0.05, 1.0)
# sub drone
bed += 0.16 * np.sin(2 * np.pi * mtof(26) * t) * (0.6 + 0.4 * env(st, 6, END, 3, 1))
# airy noise bed
air = lp_fast(hp_fast(rng.standard_normal(n), 3000), 9000)
bed += 0.012 * air * (0.6 + 1.5 * env(st, 12, 17, 4, 0.1))

# crossfade the overhang onto the start -> seamless loop
X = int(XF * SR); N = int(T * SR)
fade = np.linspace(0, 1, X)
bed_out = bed[:, :N].copy()
bed_out[:, :X] = bed[:, :X] * np.sqrt(fade) + bed[:, N:N + X] * np.sqrt(1 - fade)

# ---------------- events ----------------
ev = np.zeros((2, n))
dry = np.zeros(n)

def add(sig, pan=0.0, rev=0.0, gain=1.0):
    global ev, dry
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    ev[0] += sig * l * gain * 1.41; ev[1] += sig * r * gain * 1.41
    dry += sig * rev * gain

def hit(at, amp=1.0, f0=70, f1=32, dur=1.4):
    i0 = int((at + O) * SR); m = int(dur * SR); k = np.arange(m) / SR
    f = f1 + (f0 - f1) * np.exp(-k * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 2.8)
    click = rng.standard_normal(m) * np.exp(-k * 60) * 0.5
    y = np.zeros(n); seg = (s + lp_fast(click, 4000)) * amp; y[i0:i0 + m] = seg[: n - i0]
    return y

def whoosh(a, b, f0, f1, amp=0.3, rev=False):
    i0, i1 = int((a + O) * SR), int((b + O) * SR); m = i1 - i0; k = np.linspace(0, 1, m)
    no = rng.standard_normal(m)
    X = np.fft.rfft(no); f = np.fft.rfftfreq(m, 1 / SR)
    # sweep approximated in 8 band slices
    y = np.zeros(m)
    for q in range(8):
        a0, a1 = q / 8, (q + 1) / 8
        fc = f0 * (f1 / f0) ** ((a0 + a1) / 2)
        band = np.fft.irfft(X * np.exp(-((np.log(f + 1) - np.log(fc)) ** 2) / 0.35), m)
        w = np.clip(1 - np.abs(k - (a0 + a1) / 2) * 8 / 1.5, 0, 1)
        y += band * w
    e = k ** 2 if not rev else k ** 3
    y = y / (np.abs(y).max() + 1e-9) * e * amp
    out = np.zeros(n); out[i0:i1] = y
    return out

def blip(at, f=1200, amp=0.12, glide=1.0):
    m = int(0.18 * SR); k = np.arange(m) / SR
    ff = f * (1 + (glide - 1) * k / 0.18)
    s = np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-k * 28)
    y = np.zeros(n); i0 = int((at + O) * SR); y[i0:i0 + m] = s * amp
    return y

def rewind_fx(a, b, amp):
    """tape-rewind: descending noise sweep + falling tone swell over [a, b] (story time)"""
    add(whoosh(a, b, 6000, 300, amp, rev=True), rev=0.6)
    sw = np.zeros(n); i0 = int((a + 0.05 + O) * SR); i1 = int((b + O) * SR); kk = np.arange(i1 - i0) / (i1 - i0)
    sw[i0:i1] = np.sin(2 * np.pi * np.cumsum(mtof(74) * (1 - 0.5 * kk)) / SR) * kk ** 3 * 0.06
    add(sw, rev=0.8)

if COLD:
    # cut from the chaos flash into the rewind: hit + air burst, rewind sweep, question stinger
    add(hit(-0.5, 0.95, 95, 36, 1.6), rev=0.5)
    cb = np.zeros(n); i0 = int((O - 0.5) * SR); m = int(0.4 * SR); kk = np.arange(m) / SR
    cb[i0:i0 + m] = lp_fast(rng.standard_normal(m), 7000) * np.exp(-kk * 12) * 0.3
    add(cb, rev=0.6)
    rewind_fx(-0.5, 0.0, 0.5)
    for m, d in zip([62, 69, 74, 77], [0, 0.03, 0.06, 0.09]):
        add(note(mtof(m), -0.45 + d, 2.2, 0.07, "bell"), pan=(m - 70) / 12, rev=0.8)
    add(blip(-0.3, 900, 0.08, 1.6), pan=0.2, rev=0.3)
    add(hit(0.02, 0.5, 80, 38, 1.0), rev=0.3)          # release from rest
else:
    # hook: hit + bright chord stinger at the loop start (and again via the rewind lead-in)
    add(hit(0.02, 0.9, 90, 38, 1.6), rev=0.4)
    for m, d in zip([62, 69, 74, 77], [0, 0.03, 0.06, 0.09]):
        add(note(mtof(m), 0.1 + d, 2.0, 0.07, "bell"), pan=(m - 70) / 12, rev=0.8)
    add(blip(0.32, 900, 0.08, 1.6), pan=0.2, rev=0.3)
# text pops
for at, f in [(2.25, 1500), (2.55, 1100), (4.3, 1300), (6.15, 1000), (9.2, 1700), (11.55, 1000), (18.2, 1500), (19.1, 1700)] + ([] if COLD else [(22.35, 2200)]):
    add(blip(at, f, 0.07, 1.25), pan=0.15, rev=0.5)
# "It's 1,000." punch + zoom-lens glide
add(hit(8.7, 0.55, 120, 50, 0.8), rev=0.3)
add(blip(8.85, 400, 0.10, 6.0), pan=-0.4, rev=0.6)
# pendulum tick pulse (the clock of determinism): 2.0 -> 16.9 s, quarter notes at 100 bpm
beat = 60 / 100
k = 0
while 2.0 + k * beat < 16.9:
    at = 2.0 + k * beat
    tick = blip(at, 3200 if k % 2 else 2600, 0.05 + 0.03 * env(np.array([at]), 6, 17, 6, 0.1)[0], 0.98)
    add(tick, pan=0.5 if k % 2 else -0.5, rev=0.25)
    k += 1
# heartbeat under "...for now" (accelerating)
hb = 14.1
while hb < 16.9:
    add(hit(hb, 0.45, 60, 40, 0.35)); add(hit(hb + 0.18, 0.3, 55, 38, 0.3))
    hb += 0.75 - 0.18 * (hb - 14.1) / 2.8
# riser into the reveal
add(whoosh(12.5, 17.0, 200, 7000, 0.22), rev=0.5)
r = np.zeros(n); i0, i1 = int((14.0 + O) * SR), int((17.0 + O) * SR); kk = np.arange(i1 - i0) / SR
r[i0:i1] = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (kk / 3.0 * 2)) / SR) * (kk / 3.0) ** 2 * 0.08
add(r, rev=0.4)
# REVEAL: sub drop + noise burst + chord shimmer
add(hit(17.0, 1.25, 110, 28, 2.6), rev=0.5)
burst = np.zeros(n); i0 = int((17.0 + O) * SR); m = int(1.5 * SR); kk = np.arange(m) / SR
burst[i0:i0 + m] = lp_fast(rng.standard_normal(m), 6000) * np.exp(-kk * 4) * 0.35
add(burst, rev=0.9)
for m, d in zip([50, 57, 62, 65, 69, 74, 76], np.arange(7) * 0.025):
    add(note(mtof(m), 17.02 + d, 3.0, 0.09, "bell"), pan=(m - 62) / 16, rev=1.0)
# sparkle arpeggio 17.5 -> 22.2 (D minor add9, 16ths at 100 bpm)
arp = [62, 65, 69, 74, 76, 74, 69, 65]
step = beat / 4; k = 0; at = 17.45
ARP_END = 22.7 if COLD else 22.1   # cold open: keeps playing through the wrapped 0.5 s flash
while at < ARP_END:
    m = arp[k % len(arp)] + (12 if (k // 8) % 2 else 0)
    a = 0.075 * (0.7 + 0.3 * ((k % 4) == 0)) * min(1, (at - 17.4) / 0.6) * (1 if COLD else min(1, (22.2 - at) / 0.8))
    add(note(mtof(m), at, 0.5, a, "pluck"), pan=np.sin(k * 0.7) * 0.6, rev=0.7)
    k += 1; at += step
# "CHAOS" title thump
add(hit(17.05, 0.4, 200, 80, 0.4))
# REWIND: tape-stop sweep + reverse swell into the loop point
if not COLD:
    rewind_fx(22.2, 24.0, 0.42)

wet = reverb(dry, 2.8, 3.0)
mix = bed_out + ev[:, :N] + 0.35 * wet[:, :N]
# the reverb/event overhang past 24 s wraps onto the start as well
over = (ev[:, N:] + 0.35 * wet[:, N:])
mix[:, :over.shape[1]] += over
mix = np.tanh(mix * 0.9) / 0.9
mix /= np.abs(mix).max() / 0.89
pcm = (mix.T * 32767).astype(np.int16)
OUT = "score_cold.wav" if COLD else "score.wav"
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(OUT, pcm.shape[0] / SR, "s")
