#!/usr/bin/env bash
# Full pipeline for SimTheory #05. Deterministic: same inputs -> same frames.
# deps: python3 numpy scipy skia-python pyloudnorm mpmath pillow, ffmpeg (libx264), Inter + JetBrains Mono fonts
set -euo pipefail
cd "$(dirname "$0")"
python3 sim.py                              # 1. physics + checks -> data/sim.npz, data/measurements.json (fails if a check fails)
python3 audio.py                            # 2. soundtrack from the same time map -> out/mix.wav, data/audio_cues.json
python3 render.py video out/master.mkv      # 3. 2520 frames, offline, lossless master (FFV1 RGB) + data/frame_log.json
bash encode.sh                              # 4. main MP4 (1440x2560) + Instagram MP4 (1080x1920)
python3 cover.py                            # 5. cover PNG
python3 qa.py                               # 6. checks, key stills, measurement_log.md
