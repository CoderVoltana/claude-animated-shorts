# SimTheory #07 - Can a computer predict chaos? - measurement log

All numbers below are produced by `physics/run_sims.py` + `physics/analyze.py` and read by the renderer and audio from `data/measurements.json`; nothing on screen is typed by hand.

## Setup

- Double pendulum, equal arms 1 m, equal masses 1 kg, g = 9.81 m/s^2, no friction, released from rest at theta1 = theta2 = 120 deg.
- Identical code for every run (`physics/dp.py`): same equations, classic RK4, dt = 1 ms, same start angles. Only the number type changes. Every constant (g, dt, 1/6, pi, 120 deg) is converted in the run's own precision.
- float32 run: every operation in numpy float32 (type asserted on the final state). float64: Python floats. Arbitrary precision: mpmath with a mantissa of round(digits x log2 10) bits.
- Simulated 120 s of sim time per run (120000 steps), every state saved to disk.
- Prediction horizon = first 1 ms step at which the bottom-arm angle theta2 differs from the reference by more than 1 deg.

## Prediction horizons

| Run | Digits (label) | Mantissa bits | Exact digits | Horizon (s, sim) | Split in video (s) |
|---|---|---|---|---|---|
| float32 (single) | 7 | 24 | 7.22 | **9.696** | 7.958 |
| float64 (double) | 16 | 53 | 15.95 | **23.069** | 18.245 |
| arbitrary, 24 digits | 24 | 80 | 24.08 | **35.616** | chart only |
| arbitrary, 32 digits | 32 | 106 | 31.91 | **46.622** | chart only |
| arbitrary, 40 digits | 40 | 133 | 40.04 | **59.801** | chart only |

## Fit: horizon = a + b x digits

- **Slope b = 1.5092 s per digit** (shown as "+1.5 s per digit"), intercept a = -0.957 s, R^2 = 0.99947.
- Residuals (s): +0.09, -0.12, +0.35, -0.71, +0.39.
- **Digits needed for a 60 s prediction: 40.39** (shown as "~40 digits"). Consistent with the direct 40-digit measurement of 59.80 s.
- Cross-check fitting on the exact digits each format keeps: slope 1.5173 s/digit, 60 s needs 40.34 digits.

## Lyapunov exponent (estimate)

- From the fit slope: lambda = ln(10) / b = **1.526 1/s** (error grows x10 every 1.51 s, e-folding time 0.66 s).
- Cross-check, log-slope of the error envelope of each run against the reference: f32 1.13, f64 1.52, mp24 1.29, mp32 1.28, mp40 1.32 1/s (cruder: it includes the slow start before errors are amplified).

## Reference precision and its check

- Reference precision used: **100 digits** (332-bit mantissa).
- Check run: 120 digits (399 bits). Max |difference| over all four state variables at every 1 ms step over the chart window (0-59.801 s): **1.52e-61** (requirement < 1e-20: PASS). Over the full 120 s: 2.46e-27. No increase of the reference precision was needed.

## Energy drift (max |E - E0| / |E0|, E0 = 14.715 J, sampled every 10 ms)

- f32: 4.76e-04
- f64: 1.99e-08
- mp24: 2.16e-08
- mp32: 1.76e-08
- mp40: 2.19e-08
- mp100: 1.72e-08  <- reference, requirement < 0.1 %: PASS
- mp120: 1.72e-08

All runs share the same RK4 truncation error (~2e-8); the float32 run's larger drift (4.8e-4) is its own rounding error.

## Playback and timing

- Release at 0.5 s video time; one constant playback speed **x1.3** (sim time = (video time - 0.5) x 1.3) until the rewind at 33.0 s.
- 7-digit split at **7.96 s** video, 16-digit split at **18.25 s** video.
- Note: the measured horizons have a ratio of 2.38, so no constant speed can put the 7-digit split at 11-13 s and the 16-digit split at 17-19 s at the same time (that needs a ratio near 1.5). x1.3 keeps the 16-digit split in its window before the chart at 21 s; the 7-digit split arrives earlier, at 7.96 s, and the captions are timed from the measured mapping.
- Rewind 33.0-35.7 s replays the recorded states backwards from sim 42.25 s to 0; the last frame is the first frame.

## Captions as shown (from the timeline)

| Start | End | Text |
|---|---|---|
| 0.00 | 2.50 | Can a computer predict chaos? (hook) |
| 2.50 | 4.90 | Same physics. Same start. Same code. |
| 4.90 | 7.88 | Only difference: digits the computer keeps. |
| 8.01 | 13.00 | 7 digits: wrong after 9.7 seconds. |
| 18.30 | 21.00 | 16 digits: wrong after 23.1 seconds. |
| 21.50 | 25.50 | Every extra digit buys the same small slice. |
| 25.50 | 29.00 | To predict 1 minute ahead: ~40 digits. |
| 29.00 | 33.00 | Real measurements are far less precise. |

Chart labels: points 9.7 / 23.1 / 35.6 / 46.6 / 59.8 s; "+1.5 s per digit"; gold dashed line at 60 s with a ring at 40.4 digits; magenta band = fewer than 6 digits (illustrative, labelled "real measurements", no number claimed).

## Audio

- Integrated loudness **-14.0 LUFS**, true peak **-5.0 dBTP**, LRA 9.5 LU (ffmpeg ebur128 on the master WAV). Drone bed set 9 dB under the effects. High-pass 40 Hz, low-pass 11 kHz, exact zero at the loop seam.

## Delivery QA (measured on the encoded files)

- main: 1440x2560, 60/1 fps, 2160 frames, video 36.000 s, h264 High yuv420p, 49.86 Mbps, aac 48000 Hz 285 kbps, faststart True, 225.7 MB, loudness -14.0 LUFS / -5.0 dBTP.
- instagram: 1080x1920, 60/1 fps, 2160 frames, video 36.000 s, h264 High yuv420p, 39.89 Mbps, aac 48000 Hz 285 kbps, faststart True, 180.8 MB, loudness -14.0 LUFS / -5.0 dBTP.
- Loop: rendered first and last frames identical = True; encoded max pixel difference 6, PSNR 67.8 dB.
- Motion: first changed frame 1 (pulse + slow push on the deliberate hold), release at frame 30; pixel-identical consecutive frame pairs: 0 (longest run 0).
- Safe-area violations by any text: 0.
- On-screen numbers vs this log: 7-digit horizon (caption) OK, 16-digit horizon (caption) OK, digits for 60 s (caption) OK, slope label s/digit OK, HUD speed OK, chart point 7 digits OK, chart point 16 digits OK, chart point 24 digits OK, chart point 32 digits OK, chart point 40 digits OK.
