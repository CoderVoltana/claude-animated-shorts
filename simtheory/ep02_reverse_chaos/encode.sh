#!/bin/bash
# Final encode: lossless render + mix.wav -> 2-pass 40 Mb/s H.264 High (yuv420p, BT.709 tv range) + AAC 320k,
# 1440x2560 @ 60 fps, faststart.
cd "$(dirname "$0")"
OUT=build/simtheory_02_reverse_chaos.mp4
VF="scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p"
X264=(-c:v libx264 -preset slow -profile:v high -level:v 5.2 -b:v 40M -maxrate 60M -bufsize 80M -g 30 -bf 2
      -x264-params "aq-mode=3:deblock=-1,-1" -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv)
ffmpeg -y -loglevel error -i build/video_lossless.mkv -vf "$VF" "${X264[@]}" -pass 1 -passlogfile build/x264 -an -f null /dev/null
ffmpeg -y -loglevel error -i build/video_lossless.mkv -i build/mix.wav -map 0:v:0 -map 1:a:0 -vf "$VF" "${X264[@]}" \
  -pass 2 -passlogfile build/x264 -r 60 -c:a aac -b:a 320k -ar 48000 -t 30 -movflags +faststart "$OUT"
ls -la "$OUT"
