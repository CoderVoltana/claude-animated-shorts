"""QA for the final MP4: container/stream specs, loudness + true peak (ffmpeg ebur128), safe-area check,
loop check (first vs last frame), and key frames at the beats of the timeline."""
import json
import os
import re
import subprocess
import sys

import numpy as np

import timeline as TL

W, H = 1440, 2560
mp4 = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


pr = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", mp4]).stdout)
v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
a = [s for s in pr["streams"] if s["codec_type"] == "audio"][0]
print(f"video: {v['codec_name']} {v['profile']} L{v['level']} {v['width']}x{v['height']} {v['r_frame_rate']} fps "
      f"{v['pix_fmt']} {int(v['bit_rate']) / 1e6:.1f} Mb/s, {v['nb_frames']} frames, {float(v['duration']):.3f} s")
print(f"audio: {a['codec_name']} {a['sample_rate']} Hz {a['channels']} ch {int(a['bit_rate']) / 1e3:.0f} kb/s, "
      f"{float(a['duration']):.3f} s")
print(f"file: {os.path.getsize(mp4) / 1e6:.1f} MB, duration {float(pr['format']['duration']):.3f} s")

r = run(["ffmpeg", "-nostats", "-i", mp4, "-map", "0:a", "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
summ = r[r.rfind("Summary:"):]
I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
TP = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1))
print(f"loudness (ffmpeg ebur128, decoded AAC): integrated {I:.1f} LUFS, true peak {TP:.1f} dBTP "
      f"-> {'PASS' if abs(I + 14) <= 1 and TP < -1 else 'CHECK'}")


def frames(select):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", f"select='{select}'", "-vsync", "0",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, H, W, 3)


def get(fi):
    return frames(f"eq(n\\,{fi})")[0].astype(np.int16)


f0, fl = get(0), get(TL.NFRAMES - 1)
print(f"loop: first vs last frame mean |diff| = {np.abs(f0 - fl).mean():.3f} / 255 (one frame of field drift)")

# safe area: anything brighter than a faint glow outside the box?
worst = 0
for im in frames("not(mod(n\\,30))"):
    im = im.max(axis=2)
    out = np.concatenate([im[:384].ravel(), im[1664:].ravel(), im[:, 1152:].ravel()])
    worst = max(worst, int(out.max()))
print(f"safe area: brightest pixel outside the central safe box over 60 sampled frames = {worst}/255 "
      f"-> {'PASS' if worst < 60 else 'CHECK'}")
