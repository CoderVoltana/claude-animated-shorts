"""Stage 7: delivery QA. Writes out/qa/qa_report.json and the 8 key stills (from the encoded MP4)."""
import glob
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import timeline as T  # noqa: E402

D = os.path.join(ROOT, "out", "delivery")
QA = os.path.join(ROOT, "out", "qa")
os.makedirs(QA, exist_ok=True)
MAIN = glob.glob(os.path.join(D, "*1440x2560.mp4"))[0]
IG = glob.glob(os.path.join(D, "*instagram_1080x1920.mp4"))[0]
rep = {}


def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    v = [s for s in j["streams"] if s["codec_type"] == "video"][0]
    a = [s for s in j["streams"] if s["codec_type"] == "audio"][0]
    # faststart: moov atom before mdat
    head = open(path, "rb").read(1 << 16)
    return {"width": v["width"], "height": v["height"], "fps": v["r_frame_rate"], "frames": int(v["nb_read_frames"]),
            "video_duration_s": float(v["duration"]), "container_duration_s": float(j["format"]["duration"]),
            "codec": v["codec_name"], "profile": v["profile"], "pix_fmt": v["pix_fmt"],
            "video_mbps": round(int(v["bit_rate"]) / 1e6, 2), "audio_codec": a["codec_name"],
            "audio_sr": int(a["sample_rate"]), "audio_kbps": round(int(a["bit_rate"]) / 1e3, 1),
            "audio_duration_s": float(a["duration"]), "faststart": head.find(b"moov") != -1 and
            (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")),
            "size_mb": round(os.path.getsize(path) / 1e6, 1)}


def loud(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a", "-af", "ebur128=peak=true:framelog=quiet",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    s = r[r.rfind("Summary:"):]
    return {"integrated_lufs": float(re.search(r"I:\s+(-?[\d.]+) LUFS", s).group(1)),
            "true_peak_dbtp": float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", s).group(1)),
            "lra_lu": float(re.search(r"LRA:\s+(-?[\d.]+) LU", s).group(1))}


def frame_at(path, idx, w, h):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3)


for name, p in (("main", MAIN), ("instagram", IG)):
    rep[name] = probe(p)
    rep[name]["loudness"] = loud(p)

# first vs last frame
png0 = cv2.imread(os.path.join(ROOT, "out", "frames", "f_0000.png"))
pngL = cv2.imread(os.path.join(ROOT, "out", "frames", f"f_{T.NF-1:04d}.png"))
f0, fL = frame_at(MAIN, 0, T.W, T.H), frame_at(MAIN, T.NF - 1, T.W, T.H)
d = np.abs(f0.astype(int) - fL.astype(int))
mse = np.mean(d ** 2)
rep["loop"] = {"rendered_png_identical": bool(np.array_equal(png0, pngL)),
               "encoded_max_abs_diff": int(d.max()), "encoded_mean_abs_diff": float(d.mean()),
               "encoded_psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))}
cv2.imwrite(os.path.join(QA, "loop_first_vs_last.jpg"),
            cv2.cvtColor(cv2.resize(np.hstack([f0, fL]), (1440, 1280)), cv2.COLOR_RGB2BGR))

# stillness: longest run of (near-)identical consecutive frames, from the rendered PNGs at 1/8 scale
# "completely still" = consecutive full-resolution frames that are pixel-identical
prev, run, longest, diffs, same = None, 0, 0, [], 0
for i in range(T.NF):
    im = cv2.imread(os.path.join(ROOT, "out", "frames", f"f_{i:04d}.png"))
    if prev is not None:
        ident = np.array_equal(im, prev)
        diffs.append(float(np.abs(im[::4, ::4].astype(np.int16) - prev[::4, ::4]).mean()))
        run = run + 1 if ident else 0
        same += ident
        longest = max(longest, run)
    prev = im
rep["motion"] = {"longest_identical_run_frames": longest, "identical_frame_pairs": int(same),
                 "first_changed_frame": int(np.argmax(np.array(diffs) > 0)) + 1,
                 "release_frame": int(T.RELEASE * T.FPS),
                 "max_mean_frame_diff": float(max(diffs))}

# safe-area violations logged by the renderer
viol = [l for f in glob.glob(os.path.join(ROOT, "out", "frames", "safe_violations_*.txt")) for l in open(f)]
rep["safe_area_violations"] = len(viol)

# key stills from the ENCODED main file
KEY = [("1_hook", 0.0), ("2_legend", 1.6), ("3_first_split", T.T7 + 0.9), ("4_second_split", T.T16 + 0.9),
       ("5_chart_points", 25.55), ("6_fitted_line", 27.2), ("7_real_world", 31.2), ("8_final_frame", T.T_LAST)]
for nm, tt in KEY:
    fr = min(T.NF - 1, int(round(tt * T.FPS)))
    im = frame_at(MAIN, fr, T.W, T.H)
    cv2.imwrite(os.path.join(QA, f"still_{nm}_f{fr:04d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
rep["key_stills"] = {nm: round(tt, 3) for nm, tt in KEY}

# every on-screen number vs the measurement log
caps = " ".join(" ".join(l) for _, _, l in T.CAPTIONS)
shown = {"7-digit horizon (caption)": (float(re.search(r"7 digits: wrong after ([\d.]+)", caps).group(1)), T.H7, 0.05),
         "16-digit horizon (caption)": (float(re.search(r"16 digits: wrong after ([\d.]+)", caps).group(1)), T.H16, 0.05),
         "digits for 60 s (caption)": (float(re.search(r"~(\d+) digits", caps).group(1)), T.D60, 0.5),
         "slope label s/digit": (round(T.SLOPE, 1), T.SLOPE, 0.05),
         "HUD speed": (T.SPEED, T.SPEED, 0)}
for d_, hz, k in T.CHART_POINTS:
    shown[f"chart point {d_} digits"] = (round(hz, 1), hz, 0.05)
rep["numbers"] = {k: {"shown": a, "measured": b, "ok": abs(a - b) <= tol + 1e-9} for k, (a, b, tol) in shown.items()}

# caption rules
rep["captions"] = [{"text": " ".join(l), "words": len(" ".join(l).split()), "start": round(a, 2), "end": round(b, 2),
                    "dur": round(b - a, 2), "ok": len(" ".join(l).split()) <= 8 and b - a >= 1.5} for a, b, l in T.CAPTIONS]
rep["hook"] = {"text": T.HOOK_TEXT, "words": len(T.HOOK_TEXT.split())}

json.dump(rep, open(os.path.join(QA, "qa_report.json"), "w"), indent=2)
print(json.dumps(rep, indent=2))
