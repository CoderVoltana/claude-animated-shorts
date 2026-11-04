# SimTheory #06: Does length change everything? Measurement log

All values below are produced by `sim.py` (physics) and `qa.py` (checks). Every on-screen number is read from the same files.

## Setup

- Model: double pendulum, m1=m2=1 kg, g=9.81, no friction, RK4 float64. Released from rest at θ1 = θ2 = 90° (both arms horizontal), video time 0.5 s, played at 1× real time.
- Timestep **dt = 0.4167 ms** (1/2400 s), RK4, float64. States saved at 2400 Hz (40 per video frame).
- Chaos clock: invisible twin with θ1 + 1e-09 rad. Clock stops when the two bottom-arm angles differ by more than 1.0°. Horizon 60 s of sim time.
- Flip: the bottom arm passes straight up and then swings down past horizontal on the far side. A wobble over the top that falls back the way it came does not count.
- Energy reference: potential measured from the lowest configuration (both bobs hanging straight down), so E0 > 0 and relative drift is well defined.

## Results

| Ratio (upper:lower) | Arms (m) | Chaos clock | Chaos clock at dt/2 | Flips by 16.5 s (end of painting) | Final flips (sim 28.5 s, shown at 29.0 s) | Max energy drift (60 s) |
|---|---|---|---|---|---|---|
| 1:3 | 0.5 + 1.5 | **not within 60 s** | not within 60 s | 0 | 0 | 2.16e-11 (2.2e-09 %) |
| 1:1 | 1.0 + 1.0 | **22.5 s** (22.4558 s) | 22.4558 s | 2 | 6 | 4.60e-11 (4.6e-09 %) |
| 3:1 | 1.5 + 0.5 | **21.0 s** (20.9904 s) | 20.9904 s | 11 | 19 | 1.77e-10 (1.8e-08 %) |
| 9:1 | 1.8 + 0.2 | **12.6 s** (12.5950 s) | 12.5948 s | 7 | 12 | 1.24e-10 (1.2e-08 %) |

**Final ranking (shortest time until unpredictable first):** 9:1 (12.6 s) → 3:1 (21.0 s) → 1:1 (22.5 s) → 1:3 (not within 60 s)
- Most chaotic: **9:1**. Most predictable: **1:3**. These are the captions in the video.

## Physics checks

- Energy drift < 0.1 % for every pendulum: **PASS** (worst 1.8e-10, i.e. about 10⁻⁸ %).
- Timestep-halving check (max distance between bottom-bob paths at dt and dt/2 over the first 10 s < 0.5 px at the largest on-screen scale (260 px/m)): **PASS** at dt = 0.4167 ms.

| Ratio | Max bottom-bob difference, dt vs dt/2, first 10 s | In pixels at 260 px/m | dt and dt/2 runs visually identical until |
|---|---|---|---|
| 1:3 | 3.58e-11 m | 9.32e-09 px | 28.50 s |
| 1:1 | 2.82e-11 m | 7.33e-09 px | 27.07 s |
| 3:1 | 2.38e-08 m | 6.18e-06 px | 26.64 s |
| 9:1 | 1.79e-07 m | 4.66e-05 px | 15.32 s |

The last column is how long the dt and dt/2 runs stay within half a pixel over the full 28.5 s shown. Past that point they drift apart only because chaos magnifies round-off at the 1e-15 level. That is the same sensitivity the chaos clock measures, not a timestep error. The chaos-clock times themselves agree at dt/2 to within 0.2 ms.
- Upward reach (needed for layout): energy conservation limits both bobs to min(L1, L2/2) ≤ 0.5 m above the pivot. Measured maxima: 1:3 0.16 m, 1:1 0.45 m, 3:1 0.23 m, 9:1 0.05 m.

## Video and audio checks

- Main: 1440×2560, 60/1 fps, 1920 frames, 32.000 s, h264 High, yuv420p, 49.8 Mbps video. Audio: aac 48000 Hz, encoded at 320 kbps (ffmpeg native AAC, twoloop coder; measured average 287 kbps because the encoder does not spend the full budget on this soft material), 2 ch. Atom order: type:'moov' then type:'mdat' (faststart).
- Instagram: 1080×1920, 60/1 fps, 1920 frames, 39.9 Mbps.
- Loudness (ffmpeg ebur128, main MP4): **-14.0 LUFS integrated**, **-1.7 dBTP true peak**, LRA 2.4 LU. ffmpeg loudnorm measurement: I = -13.9 LUFS, TP = -1.68 dBTP. Instagram copy: -14.0 LUFS, -1.7 dBTP.
- Effects vs bed: bed set 9.0 dB under the effects (BS.1770 integrated). High-pass 40 Hz (4th order), low-pass 12 kHz, −2 dB shelf above 6 kHz.
- Loop seam: first and last audio samples = 0.0 / 0.0. RMS of the last 50 ms -130.3 dBFS, first 50 ms -47.5 dBFS (no click).
- First vs last frame: lossless master **bit-identical = True**. Main MP4 decoded max pixel difference 4 (PSNR 70.56 dB). Instagram 5 (PSNR 67.51 dB).
- Safe area (x 86–1152, y 384–1664): every text, label, counter and chart pixel of all 1920 frames lies in the union box [86, 399, 1151, 1663]. Frames outside: **0**.
- Pendulums stay inside their cells and below the labels for the whole run: 1:3 yes, 1:1 yes, 3:1 yes, 9:1 yes. Same scale for all four: 120 px/m (overlay 240 px/m).
- On-screen numbers match this log: **True**. Chart: {'9:1': '12.6 s', '3:1': '21.0 s', '1:1': '22.5 s', '1:3': 'not within 60 s'}. Counters at 29.0 s: {'1:3': 0, '1:1': 6, '3:1': 19, '9:1': 12}. Counters on the last frame: {'1:3': 0, '1:1': 0, '3:1': 0, '9:1': 0}.
- Text sizes: hook 130 px, captions 96 px, labels/counters/chart values 44 px, chart labels 48 px, chart title 60 px, SIM #06 44 px (Inter Light). Lines: trails ≥ 3.2 px, rods 4 px, hairlines 3 px.

## Timeline (video time → sim time)

- 0.0–0.5 s: hook hold, sim 0. 0.5–29.0 s: sim = t − 0.5 (1×). 29.0–31.75 s: rewind sim 28.5 → 0 (smootherstep). 31.75–32.0 s: frame-1 state.
- Captions: "Does length change everything?" 0.0–2.5 s; "Same angles. Same total length." 2.5–6.0 s; "Only the arm ratio changes." 6.0–9.5 s; "Same physics. Four personalities." 10.0–14.0 s; "Four fingerprints." 18.0–22.0 s; "Most chaotic: 9:1." 23.5–26.0 s; "Most predictable: 1:3." 26.0–29.0 s
- Sound cues (video s): flip_pop@4.8983(3:1), flip_pop@6.3037(3:1), flip_pop@7.1638(9:1), flip_pop@7.9325(3:1), flip_pop@8.8233(9:1), flip_pop@9.2821(3:1), flip_pop@9.8575(3:1), flip_pop@10.4879(3:1), flip_pop@10.5021(9:1), flip_pop@11.0633(3:1), flip_pop@11.1304(1:1), flip_pop@11.4142(9:1), flip_pop@12.3604(3:1), flip_pop@13.7238(9:1), flip_pop@14.6346(9:1), flip_pop@15.4771(1:1), flip_pop@15.6596(3:1), flip_pop@16.1496(3:1), flip_pop@16.3775(9:1), flip_pop@16.7092(3:1), flip_pop@17.0504(1:1), flip_pop@17.3062(3:1), flip_pop@17.8679(3:1), flip_pop@18.2375(1:1), flip_pop@19.1646(3:1), flip_pop@20.5767(3:1), flip_pop@21.2846(9:1), flip_pop@21.3567(1:1), flip_pop@21.9604(3:1), flip_pop@22.6533(3:1), flip_pop@23.1346(9:1), flip_pop@23.2917(3:1), flip_pop@24.7204(9:1), flip_pop@24.7971(3:1), flip_pop@25.685(1:1), flip_pop@26.4046(9:1), flip_pop@28.03(9:1), release@0.5(all), trails_start_shimmer@9.5(all), merge_whoosh_chord@17.0(all), bar_tick@23.15(9:1), bar_tick@23.6(3:1), bar_tick@24.05(1:1), bar_tick@24.5(1:3), result_accent@23.5(9:1), result_accent@26.0(1:3), reverse_swoosh@29.0(all)
