#!/bin/bash
# Render all 1800 frames in 4 parallel lossless segments, then concatenate to build/video_lossless.mkv
cd "$(dirname "$0")"; mkdir -p build
P=4; TOTAL=1800; STEP=$((TOTAL / P))
for i in $(seq 0 $((P-1))); do
  python3 render.py --seg $((i*STEP)) $(((i+1)*STEP)) build/seg$i.mkv > build/seg$i.log 2>&1 &
done
wait
: > build/segs.txt; for i in $(seq 0 $((P-1))); do echo "file 'seg$i.mkv'" >> build/segs.txt; done
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segs.txt -c copy build/video_lossless.mkv
echo done
