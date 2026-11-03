# SimTheory #02 — Can you reverse chaos?

30 s YouTube Short (1440×2560, 60 fps), drawn with the SimTheory style kit. A real double-pendulum simulation is run forward, every velocity is reversed, and three copies rewind: the GHOST (kit role *reference*, white: the recorded path played backwards), CYAN (role *main*: rounding only) and ORANGE (role *perturbed*: θ1 nudged by 10⁻¹⁰°). The velocity arrows use role *extra 1* (purple).

| File | Role |
|---|---|
| `physics.py` | equations of motion and the implicit-midpoint integrator (numba, float64) |
| `config.py` | locked tuning: θ0 = 117°, T = 24.5 s |
| `tune.py`, `sweep.txt` | the honest tuning sweep over start angle and T |
| `simulate.py` | forward run, reversal, the three copies, separations, energy and sanity checks, Lyapunov fits → `sim.npz`, `sim.json` |
| `timeline.py` | the beats, plus playback speeds solved so the measured crossings land on 16.4 s / 21.0 s |
| `render.py` | renderer built on the series style kit `simtheory-style/simtheory_style.py`: kit colour roles, glow, sizes, fonts, text styles, SIM tag, layout zones and BT.709 export (parallel `render_video`) |
| `audio.py` | synthesized soundtrack, −14 LUFS, ≤ −1 dBTP, seamless loop (uses `templates/audio/dsp.py`) |
| `qa.py` | format, safe zones, loop seam, frozen frames, loudness, contact sheet |
| `LOG.md` | every measured number |
| `METADATA.md` | title, description, hashtags, pinned comment |

Series style: `../../simtheory-style/` (fonts bundled there). Requires `pip install numpy scipy numba pillow` and ffmpeg.

Rebuild (~25 min on 4 cores):
```bash
python3 simulate.py && python3 audio.py && python3 render.py --video build/simtheory_02_reverse_chaos.mp4 --audio build/mix.wav && python3 qa.py build/simtheory_02_reverse_chaos.mp4 build/qa
```
