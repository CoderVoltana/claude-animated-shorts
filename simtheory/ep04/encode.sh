#!/usr/bin/env bash
# Step 4: encode the rendered PNG frames + synthesized audio.  Usage: ./encode.sh FRAMES_DIR OUT_DIR
# The picture is mostly black, so content-driven rate control tops out near 30 Mbps even at qp ~0.
# To meet the 40-60 Mbps delivery spec, the main file uses strict CBR (HRD, qpmin 0): the encoder spends
# what it can on quality and the rest is HRD filler.
set -euo pipefail
FR=${1:?frames dir}; OUT=${2:?out dir}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
enc () {  # $1 vf  $2 rate  $3 out
  ffmpeg -y -hide_banner -loglevel error -framerate 60 -i "$FR/%05d.png" -i "$HERE/data/audio.wav" \
    -map 0:v:0 -map 1:a:0 -vf "$1" -c:v libx264 -profile:v high -preset slow \
    -b:v "$2" -minrate "$2" -maxrate "$2" -bufsize "$2" -x264-params "nal-hrd=cbr:qpmin=0:keyint=60:min-keyint=60" \
    -r 60 $COL -t 34 -c:a aac -b:a 320k -ar 48000 -movflags +faststart "$3"
}
enc "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos,format=yuv420p" 50M "$OUT/simtheory04_friction_chaos_1440x2560.mp4"
enc "scale=1080:1920:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p" 30M "$OUT/simtheory04_friction_chaos_1080x1920_instagram.mp4"
