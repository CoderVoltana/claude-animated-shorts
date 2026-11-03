# SimTheory #02 — Can you reverse chaos?

30 s YouTube Short (1440×2560, 60 fps). A real double-pendulum simulation is run forward, every velocity is reversed, and three copies rewind: the GHOST (the recorded path played backwards), CYAN (rounding only) and ORANGE (θ1 nudged by 10⁻¹⁰°).

| File | Role |
|---|---|
| `physics.py` | equations of motion and the implicit-midpoint integrator (numba, float64) |
| `config.py` | locked tuning: θ0 = 117°, T = 24.5 s |
| `tune.py`, `sweep.txt` | the honest tuning sweep over start angle and T |
| `simulate.py` | forward run, reversal, the three copies, separations, energy and sanity checks, Lyapunov fits → `sim.npz`, `sim.json` |
| `timeline.py` | the beats, plus playback speeds solved so the measured crossings land on 16.4 s / 21.0 s |
| `render.py`, `render_all.sh` | Skia renderer (additive neon and bloom, captions, HUD, graph) → lossless segments |
| `audio.py` | synthesized soundtrack, −14 LUFS, ≤ −1 dBTP, seamless loop (uses `templates/audio/dsp.py`) |
| `encode.sh` | 2-pass 40 Mb/s H.264 High and AAC 320k MP4 |
| `qa.py` | format, safe zones, loop seam, frozen frames, loudness, contact sheet |
| `LOG.md` | every measured number |
| `METADATA.md` | title, description, hashtags, pinned comment |

Series style: `../STYLE.md`. Requires `pip install numpy scipy numba skia-python` (plus `libegl1`), ffmpeg, and the Inter fonts.

Rebuild (~5 min on 4 cores):
```bash
python3 simulate.py && python3 audio.py && ./render_all.sh && ./encode.sh && python3 qa.py build/simtheory_02_reverse_chaos.mp4 build/qa
```
