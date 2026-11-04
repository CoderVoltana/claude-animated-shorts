# SimTheory #05 · CAN A PENDULUM FLIP FOREVER?

A 42 s vertical Short (1440×2560, 60 fps). Nine pendulums share one pivot and get nine pushes (E/E_c = 0.2 … 1.7).
The trapped ones swing, the others flip forever, a phase portrait shows the boundary, and a 0.5× replay follows the
threshold pendulum (exactly E_c) as it creeps towards the top forever.

Run `./build.sh`. See `DELIVERABLES.md` for title, description, hashtags and pinned comment, and `measurement_log.md`
for every measured number.

## Pipeline
| Step | File | Output |
|---|---|---|
| Physics (RK4, float64, dt = 1/4800 s) + checks | `sim.py` | `data/sim.npz` (every state), `data/measurements.json`, `data/events_sim.json` |
| Time map (video ↔ sim clocks, captions, camera, emphasis) | `timeline.py` | used by every other step |
| Sound (synth, 48 kHz, −14 LUFS, ≤ −1 dBTP, loop-safe) | `audio.py` | `out/mix.wav`, `data/audio_cues.json` |
| Frames (Skia vector AA + multi-scale bloom) | `render.py` | `out/master.mkv` (lossless), `data/frame_log.json` |
| Encode | `encode.sh` | main + Instagram MP4 |
| Cover | `cover.py` | `out/cover_1080x1920.png` |
| QA | `qa.py` | `data/qa.json`, `measurement_log.md`, `stills/key_*.png` |

## Time map
| Video time | Clock | Sim time |
|---|---|---|
| 0.0–0.5 s | rest (hook frame) | 0 (at rest) |
| 0.5–28.0 s | live, 1× | t = tv − 0.5 |
| 27.5–28.1 s | crossfade live → replay | both |
| 28.0–38.0 s | replay, 0.5× | τ = (tv − 28)·0.5 = 0 → 5 s |
| 37.6–38.4 s | dimmed pendulums crossfade replay → live (threshold switches at 38.0 s) | both |
| 38.0–40.0 s | live, 1× | t = tv − 0.5 = 37.5 → 39.5 s |
| 40.0–41.7 s | rewind | 39.5·(1 − (1 − cos πu)/2) → 0 |
| 41.7–42.0 s | rest (identical to frame 1) | 0 |

The threshold pendulum is always drawn from θ = π − δ, δ(t) = 4·arctan(e^(−ω0·t)); the gap readout formats δ directly
(never 180° − θ), so it stays exact down to ~10⁻³⁰⁰.
