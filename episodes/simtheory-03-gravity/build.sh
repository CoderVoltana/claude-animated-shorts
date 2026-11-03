#!/usr/bin/env bash
# SimTheory #03 - full build: physics log -> KaTeX formula -> frames -> audio -> mux -> QA
set -euo pipefail
cd "$(dirname "$0")"
[ -d node_modules/katex ] || npm install --no-audit --no-fund
python3 sim.py | tee physics_log.txt
node formula.js
python3 render.py --video
python3 audio.py
mkdir -p out
ffmpeg -y -loglevel error -i build/video.mp4 -i build/audio.wav -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 320k -ar 48000 -movflags +faststart -metadata title="SimTheory #03 - What if gravity was 10x stronger?" \
  out/SimTheory_03_10x_Gravity.mp4
python3 qa.py out/SimTheory_03_10x_Gravity.mp4
