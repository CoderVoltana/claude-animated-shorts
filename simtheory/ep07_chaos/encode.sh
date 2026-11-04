#!/usr/bin/env bash
# Stage 6: encode frames + mix -> main MP4 (1440x2560) and Instagram copy (1080x1920). Both 60 fps, H.264 High, AAC 320k.
set -euo pipefail
cd "$(dirname "$0")"
OUTD=out/delivery; mkdir -p "$OUTD"
COLOR="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
enc () {  # $1 = scale filter, $2 = bitrate, $3 = output
  # constrained CBR: mostly-black frames would otherwise undershoot; x264 lowers QP to spend the budget
  ffmpeg -y -hide_banner -loglevel error -framerate 60 -i out/frames/f_%04d.png -i out/audio/mix.wav \
    -map 0:v:0 -map 1:a:0 -frames:v 2160 \
    -vf "$1,format=yuv420p" -c:v libx264 -profile:v high -preset slow -pix_fmt yuv420p \
    -b:v "$2" -minrate "$2" -maxrate "$2" -bufsize "$2" -x264-params nal-hrd=cbr -g 120 \
    $COLOR -c:a aac -b:a 320k -ar 48000 -ac 2 -shortest -movflags +faststart "$3"
}
enc "scale=1440:2560:out_color_matrix=bt709:out_range=tv:flags=lanczos" 50M "$OUTD/simtheory_ep07_can_a_computer_predict_chaos_1440x2560.mp4"
enc "scale=1080:1920:out_color_matrix=bt709:out_range=tv:flags=lanczos" 40M "$OUTD/simtheory_ep07_can_a_computer_predict_chaos_instagram_1080x1920.mp4"
