#!/usr/bin/env bash
# Step 5: encode the lossless master + synthesized mix into the delivery files.
set -euo pipefail
cd "$(dirname "$0")"
OUT=out
MAIN=$OUT/SimTheory05_Pendulum_1440x2560_60fps.mp4
IG=$OUT/SimTheory05_Pendulum_Instagram_1080x1920_60fps.mp4
COLOR="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"

# 1) main: 1440x2560, 60 fps, H.264 High, yuv420p, ~50 Mbps (2-pass), AAC 48 kHz 320 kbps, faststart
for PASS in 1 2; do
  if [ $PASS = 1 ]; then AOPT="-an"; DST="-f null /dev/null"; else AOPT="-map 1:a:0 -c:a aac -b:a 320k -ar 48000 -ac 2"; DST="-movflags +faststart $MAIN"; fi
  ffmpeg -y -v error -i $OUT/master.mkv -i $OUT/mix.wav -map 0:v:0 $AOPT \
    -vf "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p" \
    -c:v libx264 -profile:v high -preset slow -tune animation -b:v 50M -maxrate 60M -bufsize 100M \
    -pass $PASS -passlogfile $OUT/x264_main -g 120 -r 60 $COLOR -t 42 $DST
done

# 2) Instagram copy: 1080x1920, 60 fps, high bitrate
for PASS in 1 2; do
  if [ $PASS = 1 ]; then AOPT="-an"; DST="-f null /dev/null"; else AOPT="-map 1:a:0 -c:a aac -b:a 320k -ar 48000 -ac 2"; DST="-movflags +faststart $IG"; fi
  ffmpeg -y -v error -i $OUT/master.mkv -i $OUT/mix.wav -map 0:v:0 $AOPT \
    -vf "scale=1080:1920:out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p" \
    -c:v libx264 -profile:v high -preset slow -tune animation -b:v 30M -maxrate 40M -bufsize 60M \
    -pass $PASS -passlogfile $OUT/x264_ig -g 120 -r 60 $COLOR -t 42 $DST
done
rm -f $OUT/x264_*.log $OUT/x264_*.mbtree $OUT/x264_*.log.mbtree $OUT/x264_*temp* 2>/dev/null || true
ls -la $MAIN $IG
