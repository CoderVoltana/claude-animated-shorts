# SimTheory #02 — "Can you reverse chaos?" — measurement log

All numbers below come from `sim.json`, `build/audio_report.json` and `build/qa/qa.json`. Re-run with
`python3 simulate.py && python3 audio.py && ./render_all.sh && ./encode.sh && python3 qa.py build/simtheory_02_reverse_chaos.mp4 build/qa`.

## Physics
| Item | Value |
|---|---|
| System | ideal double pendulum, L1 = L2 = 1 m, m1 = m2 = 1 kg, g = 9.81 m/s², no damping |
| Integrator | implicit midpoint on Hamilton's equations (θ, p), fixed-point iterated until the iterate stops changing (≤ 8 iterations), float64, dt = 1×10⁻⁴ s. Symmetric, reversible, symplectic. p → −p is exactly ω → −ω. |
| Start | θ1 = θ2 = **117°**, at rest (tuned from 120°, see below) |
| Forward run **T** | **24.5 s** (245,000 steps, every state recorded) |
| Energy drift (max, all three runs) | **1.27×10⁻⁶** relative (limit 10⁻³) |
| Reversal sanity test (10°/5°, 5 s forward, reverse, 5 s) | returns within **3.3×10⁻¹⁶ rad** (limit 10⁻⁹): PASS |
| Hamiltonian vs textbook Lagrangian accelerations | agree to 2.5×10⁻⁹ (finite-difference limited) |
| ORANGE nudge | +10⁻¹⁰° = 1.745×10⁻¹² rad on θ1 at the reversal (stored as 1.746×10⁻¹² after float64 rounding) |
| CYAN initial error | exactly 0. The first angle difference appears 0.12 s after the reversal at 3.6×10⁻¹⁵ rad (2 ulp of θ1 ≈ 12.4 rad) |
| Ghost return error | 0 (the ghost is the recorded path) |

## Divergence (first time the angle difference to the ghost is > 2°)
| Copy | Sim time after the reversal | Fraction of the way back | Video time |
|---|---|---|---|
| ORANGE (nudge) | **10.931 s** | **44.6 %** | **16.40 s** |
| CYAN (rounding only) | **18.039 s** | **73.6 %** | **21.00 s** |
| Ghost lands at the start | 24.5 s | 100 % | 28.60 s |

## Lyapunov exponent (slope of ln(separation) vs time)
| Fit | λ (s⁻¹) | R² |
|---|---|---|
| CYAN, common window 0.5 to 9.14 s | 2.106 | 0.982 |
| ORANGE, common window 0.5 to 9.14 s | 2.108 | 0.980 |
| **Mean (shown on the graph)** | **2.107**, error doubling time 0.33 s | |
| CYAN, own window to 10⁻³ rad (0.12 to 16.0 s) | 1.859 | 0.965 |

The two curves stay parallel, about e⁸ ≈ 3,000× apart. That is roughly the ratio of the 1.7×10⁻¹² rad nudge to the ~10⁻¹⁵ rad rounding error.

## Playback (shown live in the HUD)
- Forward run: ×0.94 at the release, ramping to **×3.76**, then easing into the freeze at 9.0 s.
- Rewind: **×2.22** (11 to 16.4 s), **×1.55** (16.4 to 21 s), **×0.84** (21 to 28.6 s). The speeds were solved so that the measured crossings land on the planned beats. The visuals are not retimed in any other way.

## Tuning notes (honest deviations from the brief)
- At **120°**, no T from 22 to 50 s put ORANGE near 40 % and CYAN near 75 % together. The closest was T = 24.5 s at 70 % / 76 % (see `sweep.txt`). A 3° change to **117°** gave 44.6 % / 73.6 % at the same T = 24.5 s.
- Both copies run on the same Lyapunov clock, so their gap is fixed at ln(nudge/rounding)/λ ≈ 3.8 to 7 s of sim time. 40 % and 75 % cannot both be hit exactly. Physics wins, and the beats are 16.4 s and 21.0 s.
- The graph's CYAN curve starts from 10⁻¹⁶ (axis floor, exactly zero error) and the data proper starts at ~4×10⁻¹⁵ rad. ORANGE starts at 1.7×10⁻¹² rad (= 10⁻¹⁰°).

## Look (series style kit)
Everything is drawn with `simtheory-style/simtheory_style.py` v1.0.0: colour roles (ghost = reference white `#F4F7FF`, cyan = main `#22E5FF`, orange = perturbed `#FF8A1E`, arrows = extra 1 purple `#A848FF`, HUD/axes = `hud`), the kit's bloom and roll-off, its line/trail/bob sizes, its fonts and text styles (captions in `headline`, readouts in `hud`/`label`) and its `SIM #02` tag. Two layouts. **Instagram (square):** the kit layout scaled to 1440×2560 (×4/3): text on `text_y`, the readout and graph in `hud_band`, and the pivot placed so that the full 2 m reach stays inside the square-safe zone and left of the Shorts UI column. **YouTube (full screen):** the whole 9:16 frame minus the Shorts UI. The tag and HUD are at the top, captions in the upper-middle third, the graph in the upper-middle area, and the pendulum larger (272 px/m against 240). The sim-time counter is removed in both. The HUD shows direction and playback speed only.

Export follows the kit's colour standard (BT.709 matrix, tv range, BT.709 tags, verified with `check_export`). It differs from the kit in two documented ways, both required by the brief:
- **Bitrate.** A CRF 6 kit master is re-encoded YUV→YUV at CRF 9, about 46 Mb/s. The kit's CRF 15 gave about 5 Mb/s.
- **Audio.** The mastered mix is muxed directly. The kit's single-pass loudnorm had pushed it to −12.3 LUFS and halved its dynamic range.

## Delivery QA, Instagram / square-safe version (`build/qa_instagram/qa.json`)
| Check | Result |
|---|---|
| Video | H.264 High, 1440×2560, 60/1 fps, yuv420p, BT.709 / tv tags, 1800 frames, 30.0 s, **46.1 Mb/s** |
| Audio | AAC-LC 48 kHz stereo 256k |
| Loudness (ffmpeg ebur128) | **−14.0 LUFS** integrated, true peak **−1.7 dBTP**, LRA 8.0 LU |
| Loop seam (audio) | last→first sample jump 0.0017 (median sample step 0.0045): no click |
| Loop seam (video) | first vs last frame PSNR 50 dB, mean abs diff 0.37/255. The only difference is the kit's per-frame dither |
| Right 15 % / bottom 20 % | 0 content pixels (> 96/255) in any frame. Only the kit's nebula backdrop and faint bloom reach those zones |
| Frozen spans while moving | none |

## Delivery QA, YouTube full-screen version (`build/qa_youtube/qa.json`)
| Check | Result |
|---|---|
| Video | H.264 High, 1440×2560, 60/1 fps, yuv420p, BT.709 / tv tags, 1800 frames, 30.0 s, **42.7 Mb/s** |
| Loudness | **−14.0 LUFS**, true peak **−1.7 dBTP**, LRA 8.0 LU (same mix) |
| Loop seam (video) | PSNR 50 dB (kit dither only) |
| Right 15 % | 0 content pixels |
| Bottom 20 % | at most 56 glow pixels (> 96/255) in one frame. This is the bloom halo of the lower bob at its lowest point. The bob core stays above y ≈ 2033 |
| Frozen spans while moving | none |
