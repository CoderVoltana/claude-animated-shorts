"""Quality checks before delivery. Writes data/qa.json and the 8 key stills from the final MP4."""
import json, math, re, subprocess, sys
import numpy as np
import cv2
import skia
import render as RD
import timeline as TL

MAIN = "out/SimTheory06_DoesLengthChangeEverything_1440x2560_60fps.mp4"
IG = "out/SimTheory06_DoesLengthChangeEverything_Instagram_1080x1920_60fps.mp4"
qa = {}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


# 1. every UI/text pixel of every frame inside the safe area (render the text layer alone on transparent)
x0, y0, x1, y1 = RD.SAFE
worst = None
ui = skia.Surface.MakeRaster(skia.ImageInfo.MakeN32Premul(RD.W, RD.H))
bad = []
for i in range(0, TL.NFRAMES):
    t = TL.frame_time(i); t = 0.0 if t >= TL.REWIND_END else t
    c = ui.getCanvas(); c.clear(skia.Color4f(0, 0, 0, 0))
    RD.draw_ui(c, t, TL.sim_time(t))
    a = ui.makeImageSnapshot().toarray()[..., 3]
    ys, xs = np.nonzero(a > 8)                      # ignore the faintest anti-aliased shadow fringe
    if len(xs):
        bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if bb[0] < x0 or bb[1] < y0 or bb[2] > x1 or bb[3] > y1:
            bad.append((i, bb))
        worst = bb if worst is None else (min(worst[0], bb[0]), min(worst[1], bb[1]), max(worst[2], bb[2]), max(worst[3], bb[3]))
qa["text_ui_union_bbox_px"] = worst
qa["text_ui_outside_safe_frames"] = len(bad)
qa["text_ui_outside_examples"] = bad[:5]
print("UI bbox", worst, "frames outside safe:", len(bad))

# 2. pendulums (rods + bobs incl. 10.5 px dot) stay inside their cells for every shown sim time; never enter caption strip
geo = {}
for k, lab in enumerate(RD.LABELS):
    bx = RD.BOXES[k]; px, py = RD.PIVOTS[k]
    th = RD.TRAJ[k, : int(TL.SIM_AT_REWIND * RD.HZ) + 1]
    L1, L2 = RD.LEN[k]
    X1 = px + RD.S * L1 * np.sin(th[:, 0]); Y1 = py + RD.S * L1 * np.cos(th[:, 0])
    X2 = X1 + RD.S * L2 * np.sin(th[:, 1]); Y2 = Y1 + RD.S * L2 * np.cos(th[:, 1])
    r = 10.5
    ext = (float(min(X1.min(), X2.min()) - r), float(min(Y1.min(), Y2.min()) - r), float(max(X1.max(), X2.max()) + r), float(max(Y1.max(), Y2.max()) + r))
    label_bottom = bx[1] + 14 + RD.CAP_H["lab"]
    geo[lab] = dict(extent_px=[round(v, 1) for v in ext], cell=bx, inside_cell=bool(ext[0] >= bx[0] and ext[2] <= bx[2] and ext[1] >= bx[1] and ext[3] <= bx[3]),
                    clear_of_labels=bool(ext[1] > label_bottom), label_bottom_px=round(label_bottom, 1))
qa["pendulum_geometry"] = geo
qa["grid_scale_px_per_m"] = RD.S
print(json.dumps(geo, indent=0))

# 3. numbers on screen vs measurement log
phys = json.load(open("data/physics.json"))
onscreen = {"chart": {r["label"]: TL.fmt_chaos(r["tc"]) for r in TL.chart_rows(phys)},
            "captions": [" / ".join(c[0]) for c in TL.captions(phys)],
            "flip_counters_at_29s": {lab: RD.flip_count(k, TL.SIM_AT_REWIND) for k, lab in enumerate(RD.LABELS)},
            "flip_counters_at_frame_1919": {lab: RD.flip_count(k, TL.sim_time(0.0)) for k, lab in enumerate(RD.LABELS)}}
match = all(onscreen["chart"][l] == TL.fmt_chaos(p["chaos_clock_s"]) for l, p in phys["pendulums"].items()) and \
    all(onscreen["flip_counters_at_29s"][l] == p["flips_at_sim_28_5s_final_shown"] for l, p in phys["pendulums"].items())
qa["onscreen_numbers"] = onscreen
qa["onscreen_numbers_match_log"] = bool(match)

# 4. master: first vs last frame exactly equal (lossless)
def frames(path, idx, scale=None):
    out = []
    for i in idx:
        vf = f"select=eq(n\\,{i})" + (f",scale={scale}" if scale else "")
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vf", vf, "-vsync", "0", "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        w, h = (1440, 2560) if "1080" not in path else (1080, 1920)
        out.append(np.frombuffer(p.stdout, np.uint8).reshape(h, w, 3))
    return out

m0, m1 = frames("out/master_video.mkv", [0, 1919])
qa["master_first_last_identical"] = bool(np.array_equal(m0, m1))
for name, path in (("main", MAIN), ("instagram", IG)):
    f0, f1 = frames(path, [0, 1919])
    d = np.abs(f0.astype(int) - f1.astype(int))
    mse = (d ** 2).mean()
    qa[f"{name}_first_last_maxdiff"] = int(d.max())
    qa[f"{name}_first_last_psnr_db"] = float("inf") if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2)

# 5. container / stream checks
for name, path in (("main", MAIN), ("instagram", IG)):
    pr = json.loads(sh(f"ffprobe -v error -count_frames -show_streams -show_format -of json '{path}'").stdout)
    v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    a = [s for s in pr["streams"] if s["codec_type"] == "audio"][0]
    e = sh(f"ffmpeg -nostats -i '{path}' -map 0:a -af ebur128=peak=true -f null - 2>&1").stdout
    I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", e)[-1]); TP = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", e)[-1])
    LRA = float(re.findall(r"LRA:\s+(-?[\d.]+) LU", e)[-1])
    ln = sh(f"ffmpeg -nostats -i '{path}' -map 0:a -af loudnorm=I=-14:TP=-1:print_format=json -f null - 2>&1").stdout
    lnj = json.loads(ln[ln.rindex("{"):ln.rindex("}") + 1])
    qa[name] = dict(width=v["width"], height=v["height"], fps=v["r_frame_rate"], frames=int(v["nb_read_frames"]),
                    duration_s=float(pr["format"]["duration"]), video_codec=v["codec_name"], profile=v.get("profile"),
                    pix_fmt=v["pix_fmt"], video_kbps=round(int(v["bit_rate"]) / 1000), audio_codec=a["codec_name"],
                    audio_hz=int(a["sample_rate"]), audio_kbps=round(int(a["bit_rate"]) / 1000), channels=a["channels"],
                    ebur128_integrated_lufs=I, ebur128_true_peak_dbtp=TP, ebur128_lra_lu=LRA,
                    loudnorm_measured_I=float(lnj["input_i"]), loudnorm_measured_TP=float(lnj["input_tp"]),
                    faststart=sh(f"ffprobe -v trace -i '{path}' 2>&1 | grep -m2 -o \"type:'moov'\\|type:'mdat'\"").stdout.split())
    print(name, qa[name])

# 6. 8 key stills from the final MP4
keys = {"1_hook": 0, "2_early_motion": 90, "3_patterns_forming": 900, "4_overlay_art": 1260, "5_bar_chart": 1440,
        "6_result_most_chaotic": 1500, "7_result_most_predictable": 1680, "8_final_frame": 1919}
for nm, i in keys.items():
    f, = frames(MAIN, [i])
    cv2.imwrite(f"stills/{nm}_f{i:04d}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
qa["stills"] = keys
json.dump(qa, open("data/qa.json", "w"), indent=1, default=str)
print("qa done")
