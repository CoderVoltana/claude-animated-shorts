# SimTheory #07: Can a computer predict chaos?

A 36 s vertical short (1440x2560, 60 fps). The same double pendulum is simulated with identical code at 7, 16 and
100 digits of precision. It measures how long each copy stays within 1 deg of the reference, then shows that every extra
digit buys the same small slice of prediction time.

Results and checks are in [`MEASUREMENT_LOG.md`](MEASUREMENT_LOG.md). The title, description, hashtags and pinned comment are in [`PUBLISHING.md`](PUBLISHING.md).

## Pipeline (all offline and deterministic)

| Stage | Command | Output |
|---|---|---|
| 1. Simulate every precision (about 1.5 min on 4 cores) | `T_SIM=120 python3 physics/run_sims.py` | `data/run_*.npy`, `data/energy_*.npy`, `data/hp_mp1{00,20}.txt.gz` |
| 2. Measure (horizons, fit, Lyapunov, energy, 120-digit check) | `python3 physics/analyze.py` | `data/measurements.json` |
| 3. Render frames (cairo AA vector + HDR additive layers + bloom) | `python3 render/render.py frames <start> 2160 <step>`, one worker per core | `out/frames/f_XXXX.png` |
| 4. Synthesize audio | `python3 audio/synth.py && python3 audio/master.py` | `out/audio/mix.wav`, `loudness.json` |
| 5. Encode main + Instagram | `./encode.sh` | `out/delivery/*.mp4` |
| 6. Cover | `python3 render/render.py cover <t> delivery/cover_1080x1920.png` | `out/delivery/cover_1080x1920.png` |
| 7. QA + log | `python3 qa.py && python3 make_log.py` | `out/qa/`, `MEASUREMENT_LOG.md` |

`timeline.py` is the single source of truth. It maps video time to sim time, and it holds the caption timings, camera,
chart choreography and sound cues. Every number on screen is read from `data/measurements.json`.

Requirements: Python 3 with numpy, mpmath + gmpy2, scipy, pycairo, opencv-python-headless and Pillow (with raqm);
ffmpeg; the Inter font.
