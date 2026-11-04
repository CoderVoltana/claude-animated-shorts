#!/usr/bin/env bash
# Encode deliverables from the lossless master + synthesized audio.
set -euo pipefail
cd "$(dirname "$0")"
VF="scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
COL="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
KF="expr:eq(n,0)+eq(n,1919)"   # first and last frame are both I-frames, so they decode identically

# Mostly-black frames are cheap for x264, so plain 2-pass undershoots (~33 Mbps); strict CBR (HRD) holds the target.
enc () {  # $1 out  $2 vf  $3 constant bitrate
  ffmpeg -y -loglevel error -i out/master_video.mkv -vf "$2" -c:v libx264 -preset slow -profile:v high -b:v "$3" \
    -minrate "$3" -maxrate "$3" -bufsize "$3" -x264-params nal-hrd=cbr \
    -force_key_frames "$KF" $COL -r 60 -frames:v 1920 -pass 1 -passlogfile out/x264 -an -f null /dev/null
  ffmpeg -y -loglevel error -i out/master_video.mkv -i out/audio.wav -map 0:v -map 1:a -vf "$2" \
    -c:v libx264 -preset slow -profile:v high -b:v "$3" -minrate "$3" -maxrate "$3" -bufsize "$3" \
    -x264-params nal-hrd=cbr -force_key_frames "$KF" $COL \
    -r 60 -frames:v 1920 -pass 2 -passlogfile out/x264 \
    -c:a aac -b:a 320k -ar 48000 -t 32 -movflags +faststart "$1"
}
enc out/SimTheory06_DoesLengthChangeEverything_1440x2560_60fps.mp4 "$VF" 50M
enc out/SimTheory06_DoesLengthChangeEverything_Instagram_1080x1920_60fps.mp4 "scale=1080:1920:flags=lanczos,$VF" 40M
rm -f out/x264*
echo encoded
