# SimTheory #04 — measurement log

Energy % = (E − E_rest) / (E0 − E_rest), with E_rest = −3mgl (hanging straight down) and E0 = energy at release (14.715 J; available 44.145 J). Angle separations are wrapped to ±180° (mod 360°).

| Quantity | Value |
|---|---|
| Damping (viscous) c | 0.03 N·m·s/rad at both joints (joint 1: −c·ω1, joint 2: −c·(ω2−ω1)) |
| Dry joint friction μ (added, see note) | 0.2 N·m, smoothed as μ·tanh(ω_rel/ω0), ω0 = 0.05 rad/s |
| Twin offset (θ1, both lanes) | 0.002° |
| Start / release | θ1 = θ2 = 120° at rest, released at video 0.5 s |
| Integrator | RK4 float64, dt = 0.8333 ms (1/1200 s) |
| Playback speed | 1× from 0.5 s to 10.5 s (whole chaotic phase), smooth ramp to 7.49× (10.5–14.5 s), back to 1× (23.5–25.5 s), 1× to 31 s; labelled in the HUD at all times. Rewind 31–33.85 s. |
| Sim time shown | 0 → 108.3 s |
| Friction twins visibly split (>5° / >20°) | 5.74 s / 6.31 s video |
| No-friction twins visibly split (>5° / >20°) | 7.84 s / 8.24 s video |
| Friction lane stopped flipping (last arm over the top) | 10.19 s video (sim 9.69 s); flips per twin [3, 3] |
| Re-merge (< 0.5° apart, both arms, for good) | 21.00 s video (sim 75.64 s) |
| Rest (all speeds < 0.5°/s, for good) | 21.41 s video (sim 78.73 s) |
| Final twin separation, friction lane | 4.79e-06° (angles within 6.5e-06° of straight down) |
| Final twin separation, no-friction lane | 163.2° (bottom arm 163.2°) |
| Final energy left, no-friction lane | 99.9999998% / 99.9999998% (shown: 100%) |
| Final energy left, friction lane | 3.5e-13% / 1.6e-14% (shown: 0%) |
| Max energy drift, no-friction lane | 7.73e-07% of E0 (2.58e-07% of available energy) — limit 0.1% |
| Max energy rise per step, friction lane | 1.6e-16 (relative) — limit 1e-9 |
| Flips, no-friction lane (per twin) | [61, 55] |

## Physics checks

- PASS — no_friction_energy_drift_lt_0.1pct
- PASS — friction_energy_never_increases_lt_1e-9_per_step
- PASS — friction_twins_final_sep_lt_0.1deg
- PASS — friction_twins_hanging_down_lt_0.1deg
- PASS — split_right_gt20_by_8s
- PASS — split_left_gt20_by_9s
- PASS — flips_stop_10_12s
- PASS — remerge_20_22s
- PASS — rest_by_24s

## Delivery checks

- PASS — text_hud_inside_safe_area_all_frames
- PASS — onscreen_numbers_match_measurement_log
- PASS — first_last_rendered_frames_identical
- PASS — first_last_decoded_frames_match_main
- PASS — first_last_decoded_frames_match_instagram
- PASS — stream_specs_main
- PASS — loudness_main
- PASS — stream_specs_instagram
- PASS — loudness_instagram

## Video / audio

- Main: 1440x2560, 60/1 fps, 2040 frames, 34.000 s, H.264 High yuv420p, 49.85 Mbps video, AAC 318 kbps 48000 Hz, 213.3 MB
- Instagram: 1080x1920, 60/1 fps, 2040 frames, 29.91 Mbps video, 128.5 MB
- Loudness (ffmpeg ebur128, main MP4): integrated -14.0 LUFS, true peak -1.6 dBTP; loudnorm measure: I -14.11 LUFS, TP -1.58 dBTP
- Loudness (Instagram MP4): integrated -14.0 LUFS, true peak -1.6 dBTP
- Mix master (WAV, before AAC): -14.00 LUFS, -1.60 dBTP; loop edges |x| ≤ 2.7e-07
- Sound cues (video s): split_left 7.845, split_right 5.7408, merge_chord 20.94, rewind_swoosh 31.0
- Energy readouts shown (28–31 s): [['Energy left: 100%', 'Energy left: 0%']]

## Why there is dry friction and a labelled time-lapse (honesty note)

With the brief's friction law alone (linear viscous, −c·ω at each joint), the timing targets cannot be met at any single playback speed. Viscous joint damping acts on how fast each joint turns. Once the chaos is over, both arms swing almost together, so the remaining swing loses energy very slowly (amplitude decay ≈ 0.086·c per second). Measured: c = 0.05 keeps chaos to ~10 s, but the pendulum still holds ~9% of its energy after 30 s and needs well over 1000 s to come to rest. c ≥ 0.1 removes the chaos entirely (the twins never split at a 120° start). The time from 'chaos ends' to 'at rest' is more than 100× the chaos time, and a constant speed scales both equally.

Fix (still real physics, only the parameters were tuned, never the visuals): the viscous term is kept exactly as briefed, and each joint also gets ordinary dry (Coulomb) bearing friction μ·tanh(ω/ω0), the same at both joints and for both twins. It always opposes the motion, so energy can only fall (checked every step). Even so, settling takes ~70 s of sim time after the chaos. So playback stays 1× for the whole chaotic phase, then a smooth, always-labelled time-lapse (up to 7.5×) carries the settling, and playback returns to 1× for the final comparison. The sim clock (t = … s) and speed are on screen in every frame.
