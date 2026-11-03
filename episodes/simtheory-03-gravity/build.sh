#!/usr/bin/env bash
# SimTheory #03 - full build: physics log -> KaTeX formula -> frames (style kit) -> audio -> mux -> QA
set -euo pipefail
cd "$(dirname "$0")"
[ -d node_modules/katex ] || npm install --no-audit --no-fund
python3 sim.py > physics_log.txt
python3 -c "import timeline as TL; print(\"video time: release %.3f s, freeze/sync chord %.3f s\" % (TL.T_REL, TL.T_SYNC))" >> physics_log.txt
node formula.js
python3 render.py --video                 # kit render_video: kit export settings, BT.709 tags checked
python3 audio.py                          # mastered to the kit EXPORT targets (-14 LUFS, <= -1.5 dBTP)
mkdir -p out
# audio is already mastered and loop-continuous, so it is muxed (kit AAC settings) instead of re-loudnormed
ffmpeg -y -loglevel error -i build/video.mp4 -i build/audio.wav -map 0:v -map 1:a -c:v copy \
  -c:a aac -b:a 256k -ar 48000 -movflags +faststart \
  -metadata title="SimTheory #03 - What if gravity was 10x stronger?" out/SimTheory_03_10x_Gravity.mp4
python3 qa.py out/SimTheory_03_10x_Gravity.mp4 | tee qa_report.txt
