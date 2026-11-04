# SimTheory #06: Does length change everything?

A 32 s vertical short: four double pendulums with the same 2 m total length and the same 90°/90° release. Only the arm ratio changes (1:3, 1:1, 3:1, 9:1).

The pipeline runs in order and is fully deterministic: physics and measurements first, then an offline render from the saved states.

```bash
python3 sim.py          # RK4 sim, dt-halving check, energy drift, chaos clock, flips -> data/physics.json, data/sim.npz
python3 render.py master  # 1920 frames -> out/master_video.mkv (lossless FFV1); skia + bloom pass
python3 render.py cover   # out/SimTheory06_cover_1080x1920.png
python3 audio.py        # 48 kHz synthesized soundtrack from the same clock -> out/audio.wav
./encode.sh             # H.264 High 50 Mbps main + 30 Mbps Instagram copy, AAC 320k, faststart
python3 qa.py           # safe area, geometry, numbers-vs-log, first/last frame, loudness, stills/
python3 make_log.py     # MEASUREMENT_LOG.md, PUBLISHING.md
```

- `timeline.py` is the single clock: the video-time to sim-time map, captions, flip ticks and chart timings. The renderer and the audio both import it.
- Dependencies: numpy, scipy, numba, skia-python (needs libegl1), opencv-python-headless, ffmpeg, and the Inter fonts.
- Audio DSP helpers come from `templates/audio/dsp.py`.
