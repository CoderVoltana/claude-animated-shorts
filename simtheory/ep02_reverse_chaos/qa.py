"""QA for the final MP4: format, safe zones, loop seam, frozen spans, loudness (ffmpeg ebur128), contact sheet.

    python qa.py build/simtheory_02_reverse_chaos.mp4 build/qa
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from config import W, H, FPS, DUR


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True).stdout
    return json.loads(out)


def frames(path, w, h):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rep = {}
    pr = probe(path)
    v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    a = [s for s in pr["streams"] if s["codec_type"] == "audio"][0]
    rep["video"] = dict(color_space=v.get("color_space"), color_range=v.get("color_range"), codec=v["codec_name"], profile=v.get("profile"), w=v["width"], h=v["height"], fps=v["r_frame_rate"],
                        pix_fmt=v["pix_fmt"], frames=int(v.get("nb_frames", 0)), bitrate_mbps=int(v.get("bit_rate", 0)) / 1e6)
    rep["audio"] = dict(codec=a["codec_name"], sr=a["sample_rate"], ch=a["channels"], bitrate_kbps=int(a.get("bit_rate", 0)) / 1e3)
    rep["duration"] = float(pr["format"]["duration"])

    # full-resolution pass: safe zones, seam, motion
    safe_r, safe_b = int(W * 0.85), int(H * 0.80)
    zone_max = []
    diffs = []
    prev = None
    first = last = None
    sheet = []
    for i, f in enumerate(frames(path, W, H)):
        g = f.astype(np.int16)
        # content pixels (> 96/255 on any channel) in the right 15 % / bottom 20 %; the kit's nebula backdrop,
        # dither and faint bloom stay far below that
        zr = int(g[:safe_b, safe_r:].max())
        zb = int(g[safe_b:, :].max())
        cr = int((g[:safe_b, safe_r:].max(-1) > 96).sum())
        cb = int((g[safe_b:, :].max(-1) > 96).sum())
        zone_max.append((zr, zb, cr, cb))
        if prev is not None:
            diffs.append(float(np.abs(g[::4, ::4] - prev[::4, ::4]).mean()))
        prev = g
        if i == 0:
            first = f.copy()
        last = f
        if i % FPS == 0 or i in (int(16.4 * FPS), int(21.0 * FPS)):
            sheet.append((i, Image.fromarray(f).resize((270, 480))))
    zone_max = np.array(zone_max)
    rep["frames_decoded"] = len(zone_max)
    rep["safe_zone_right15_max_value"] = int(zone_max[:, 0].max())
    rep["safe_zone_bottom20_max_value"] = int(zone_max[:, 1].max())
    rep["safe_zone_right15_content_pixels_max"] = int(zone_max[:, 2].max())
    rep["safe_zone_bottom20_content_pixels_max"] = int(zone_max[:, 3].max())
    d = first.astype(float) - last.astype(float)
    mse = float((d ** 2).mean())
    rep["loop_first_vs_last_psnr_db"] = 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    rep["loop_first_vs_last_mean_abs"] = float(np.abs(d).mean())
    diffs = np.array(diffs)
    # frozen spans while the pendulum should be moving (forward run and rewind)
    moving = [(int(2.05 * FPS), int(8.9 * FPS)), (int(11.1 * FPS), int(28.5 * FPS))]
    frozen = []
    for a0, a1 in moving:
        seg = diffs[a0:a1]
        run = 0
        for k, x in enumerate(seg):
            run = run + 1 if x < 0.02 else 0
            if run == 6:
                frozen.append((a0 + k - 5) / FPS)
    rep["frozen_spans_during_motion"] = frozen
    rep["max_frame_jump_mean_abs"] = float(diffs.max())
    rep["max_frame_jump_at_s"] = float((np.argmax(diffs) + 1) / FPS)

    # loudness cross-check with ffmpeg
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    summ = r[r.rfind("Summary:"):]
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", summ)
    P = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ)
    LRA = re.search(r"LRA:\s+(-?[\d.]+) LU", summ)
    rep["ffmpeg_integrated_lufs"] = float(I.group(1))
    rep["ffmpeg_true_peak_dbtp"] = float(P.group(1))
    rep["ffmpeg_lra"] = float(LRA.group(1))

    cols = 8
    rows = (len(sheet) + cols - 1) // cols
    S = Image.new("RGB", (cols * 280, rows * 500), (30, 30, 30))
    for k, (i, im) in enumerate(sheet):
        dr = ImageDraw.Draw(im)
        dr.text((6, 462), f"{i / FPS:.2f}s", fill=(255, 255, 0))
        S.paste(im, ((k % cols) * 280, (k // cols) * 500))
    S.save(os.path.join(out, "contact_sheet.jpg"), quality=88)
    Image.fromarray(np.concatenate([first, last], 1)).resize((1080, 960)).save(os.path.join(out, "loop_first_last.jpg"), quality=90)
    json.dump(rep, open(os.path.join(out, "qa.json"), "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
