"""Step 5: quality checks before delivery. Usage: python qa.py FRAMES_DIR OUT_DIR
- every frame: HUD/text layer bounding box inside the safe area (x 86-1152, y 384-1664)
- every number on screen (clock, speed, energy readouts, offset caption) vs the measurement log
- first frame == last frame (rendered PNGs exactly; decoded MP4 frames by PSNR)
- container/stream checks (duration, frame count, fps, resolution, profile, bitrates) + loudness / true peak (ffmpeg)
- 8 key stills + contact sheet + cover image
"""
import csv, json, os, re, subprocess, sys
import numpy as np
import cv2
import render as RD
import timeline as TL

FR, OUT = sys.argv[1], sys.argv[2]
HERE = TL.HERE
os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
rep = {"checks": {}}
MAIN = os.path.join(OUT, "simtheory04_friction_chaos_1440x2560.mp4")
IG = os.path.join(OUT, "simtheory04_friction_chaos_1080x1920_instagram.mp4")


def check(name, ok, detail=None):
    rep["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail)


# ---------------------------------------------------------------- 1. safe area (HUD layer, every frame)
x0s, y0s, x1s, y1s = RD.SAFE
ctx = RD.Ctx()
worst = [9999, 9999, 0, 0]
bad = []
for k in range(TL.NFR):
    t = k / TL.FPS
    s = float(TL.sim_time(t))
    z, P, e = RD.camera(t)
    ef = RD.efrac_at(s)
    RD.draw_hud(ctx, t, s, e, float(ef[:2].mean()), float(ef[2:].mean()))
    a = ctx.hud[:, :, 3]
    rows = np.nonzero(a.any(axis=1))[0]
    cols = np.nonzero(a.any(axis=0))[0]
    bb = [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]
    worst = [min(worst[0], bb[0]), min(worst[1], bb[1]), max(worst[2], bb[2]), max(worst[3], bb[3])]
    if bb[0] < x0s or bb[1] < y0s or bb[2] > x1s or bb[3] > y1s:
        bad.append((k, bb))
check("text_hud_inside_safe_area_all_frames", not bad, {"union_bbox_x0y0x1y1": worst, "violations": bad[:5]})

# ---------------------------------------------------------------- 2. on-screen numbers vs measurement log
M = json.load(open(os.path.join(HERE, "data", "measure.json")))
F = list(csv.DictReader(open(os.path.join(HERE, "data", "frames.csv"))))
Hh = list(csv.DictReader(open(os.path.join(HERE, "data", "hud_log.csv"))))
mism = []
for f, h in zip(F, Hh):
    assert int(f["frame"]) == int(h["frame"])
    st = float(TL.sim_time(int(f["frame"]) / TL.FPS))
    if h["clock"] != f"t = {st:.1f} s":
        mism.append((f["frame"], "clock", h["clock"], st))
    if h["readout_left"]:
        if h["readout_left"] != f"Energy left: {int(round(float(f['energy_left_pct'])))}%":
            mism.append((f["frame"], "readL", h["readout_left"], f["energy_left_pct"]))
        if h["readout_right"] != f"Energy left: {int(round(float(f['energy_right_pct'])))}%":
            mism.append((f["frame"], "readR", h["readout_right"], f["energy_right_pct"]))
    if abs(float(h["bar_right"]) * 100 - float(f["energy_right_pct"])) > 0.05:
        mism.append((f["frame"], "barR", h["bar_right"], f["energy_right_pct"]))
caps = {h["caption"] for h in Hh if h["caption"]}
off_ok = f"Each side: two twins, {M['twin_offset_deg']:g}\u00b0 apart." in caps
check("onscreen_numbers_match_measurement_log", not mism and off_ok, {"mismatches": mism[:5], "offset_caption": off_ok})
reads = sorted({(h["readout_left"], h["readout_right"]) for h in Hh if h["readout_left"]})
rep["readouts_shown"] = reads
speeds = sorted({h["speed"] for h in Hh})
rep["speed_labels_shown"] = speeds[:3] + ["..."] + speeds[-3:]

# ---------------------------------------------------------------- 3. first frame == last frame
a = cv2.imread(os.path.join(FR, "00000.png"))
b = cv2.imread(os.path.join(FR, f"{TL.NFR - 1:05d}.png"))
check("first_last_rendered_frames_identical", np.array_equal(a, b), {"max_abs_diff": int(np.abs(a.astype(int) - b.astype(int)).max())})


def decode_frame(path, idx, w, h):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(h, w, 3)


def psnr(x, y):
    mse = np.mean((x.astype(float) - y.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


for path, (w, h), tag in ((MAIN, (1440, 2560), "main"), (IG, (1080, 1920), "instagram")):
    f0, fl = decode_frame(path, 0, w, h), decode_frame(path, TL.NFR - 1, w, h)
    p = psnr(f0, fl)
    check(f"first_last_decoded_frames_match_{tag}", p > 50, {"psnr_db": round(p, 2), "max_abs_diff": int(np.abs(f0.astype(int) - fl.astype(int)).max())})

# ---------------------------------------------------------------- 4. container / stream / loudness
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_read_frames,bit_rate,sample_rate,channels,color_space"
                          ":format=duration,bit_rate,size", "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def loud(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summ = err[err.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1))
    ln = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0", "-af",
                         "loudnorm=I=-14:TP=-1:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
    j = json.loads(ln[ln.rfind("{"):ln.rfind("}") + 1])
    return {"ebur128_integrated_lufs": I, "ebur128_true_peak_dbtp": tp,
            "loudnorm_input_i": float(j["input_i"]), "loudnorm_input_tp": float(j["input_tp"])}


for path, (w, h), tag in ((MAIN, (1440, 2560), "main"), (IG, (1080, 1920), "instagram")):
    pr = probe(path)
    v = [s_ for s_ in pr["streams"] if s_["codec_name"] == "h264"][0]
    au = [s_ for s_ in pr["streams"] if s_["codec_name"] == "aac"][0]
    dur = float(pr["format"]["duration"])
    vb = int(v["bit_rate"]) / 1e6
    ok = (v["width"], v["height"]) == (w, h) and v["r_frame_rate"] == "60/1" and int(v["nb_read_frames"]) == TL.NFR \
        and v["profile"] == "High" and v["pix_fmt"] == "yuv420p" and abs(dur - 34.0) < 0.03 and au["sample_rate"] == "48000" \
        and (40 <= vb <= 60 if tag == "main" else vb >= 20)
    check(f"stream_specs_{tag}", ok, {"res": f"{v['width']}x{v['height']}", "fps": v["r_frame_rate"], "frames": v["nb_read_frames"],
                                     "profile": v["profile"], "pix_fmt": v["pix_fmt"], "duration_s": dur, "video_mbps": round(vb, 2),
                                     "audio_kbps": round(int(au["bit_rate"]) / 1e3), "audio_sr": au["sample_rate"],
                                     "size_mb": round(int(pr["format"]["size"]) / 1e6, 1)})
    L = loud(path)
    check(f"loudness_{tag}", abs(L["ebur128_integrated_lufs"] + 14) <= 0.5 and L["ebur128_true_peak_dbtp"] <= -1.0
          and L["loudnorm_input_tp"] <= -1.0, L)
    rep[f"loudness_{tag}"] = L

# ---------------------------------------------------------------- 5. key stills, contact sheet, cover
KEY = [("01_hook", 0.0), ("02_split", 7.6), ("03_calming", 12.5), ("04_merge", M["remerge_right_video"] + 0.1),
       ("05_settled", 23.0), ("06_energy_readouts", 29.5), ("07_rewind", 32.2), ("08_final_frame", (TL.NFR - 1) / TL.FPS)]
tiles = []
for name, t in KEY:
    k = int(round(t * TL.FPS))
    img = cv2.imread(os.path.join(FR, f"{k:05d}.png"))
    cv2.imwrite(os.path.join(OUT, "stills", f"{name}_f{k:04d}.png"), img)
    tl = cv2.resize(img, (360, 640), interpolation=cv2.INTER_AREA)
    # draw the safe-area box on the thumbnail
    sc = 360 / 1440
    cv2.rectangle(tl, (int(x0s * sc), int(y0s * sc)), (int(x1s * sc), int(y1s * sc)), (0, 200, 0), 1)
    cv2.putText(tl, f"{name} t={t:.2f}", (6, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    tiles.append(tl)
sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite(os.path.join(OUT, "stills", "contact_sheet_safe_area.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
json.dump(rep, open(os.path.join(OUT, "qa_report.json"), "w"), indent=1, ensure_ascii=False)
print("ALL PASS" if all(v["pass"] for v in rep["checks"].values()) else "SOME CHECKS FAILED")
