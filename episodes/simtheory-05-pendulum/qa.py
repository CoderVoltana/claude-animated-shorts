"""Step 6: quality checks before delivery. Writes data/qa.json and measurement_log.md, exports key stills.

Independent re-checks (not reusing the renderer's own formatting):
  * every gap readout string on screen is recomputed with mpmath at 60 digits and compared
  * every text box (from the renderer's per-frame text log) must lie inside the safe area, size >= 40 px
  * first vs last frame (lossless master: exact; MP4: PSNR)
  * stillness runs, specs, loudness / true peak of the final MP4
"""
import json
import math
import os
import subprocess
import sys

import mpmath as mp
import numpy as np
from PIL import Image

import timeline as TL
from config import FRACS, H, HOOK, NFRAMES, SAFE, W, W0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MAIN = os.path.join(OUT, "SimTheory05_Pendulum_1440x2560_60fps.mp4")
IG = os.path.join(OUT, "SimTheory05_Pendulum_Instagram_1080x1920_60fps.mp4")
MASTER = os.path.join(OUT, "master.mkv")
mp.mp.dps = 60
KEY = {"hook": 0, "fan-out": 54, "trapped": 630, "flipping": 900, "phase-portrait": 1260, "boundary": 1578,
       "replay-near-top": 2190, "final-frame": NFRAMES - 1}


def run(*a):
    return subprocess.run(a, capture_output=True, text=True)


def gap_deg_mp(t):
    return mp.degrees(4 * mp.atan(mp.e ** (-mp.sqrt(mp.mpf("9.81")) * mp.mpf(repr(t)))))


def fmt_mp(d):
    """Independent formatter with the documented rules (decimals while >= 0.001 deg, then scientific)."""
    if d >= 10:
        return f"Gap to top: {float(d):.1f}°"
    if d >= 1:
        return f"Gap to top: {float(d):.2f}°"
    e = int(mp.floor(mp.log10(d)))
    if d >= mp.mpf("0.001"):
        return f"Gap to top: {float(d):.{-e + 1}f}°"
    m = d / mp.mpf(10) ** e
    if round(float(m), 1) >= 10:
        m /= 10
        e += 1
    sup = str(e).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"Gap to top: {float(m):.1f} × 10{sup} °"


def decode(path, frames, w=None, h=None):
    out = {}
    for fi in frames:
        vf = f"select=eq(n\\,{fi})" + (f",scale={w}:{h}:flags=area" if w else "")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-vsync", "0", "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        out[fi] = np.frombuffer(raw, np.uint8).reshape((h or H), (w or W), 3)
    return out


def main():
    R = {}
    meas = json.load(open(os.path.join(HERE, "data", "measurements.json")))
    flog = json.load(open(os.path.join(HERE, "data", "frame_log.json")))
    R["physics_all_pass"] = meas["all_checks_pass"]

    # ---------------- readout: every on-screen value vs high-precision recomputation
    mism, n_ro, clock_err = [], 0, 0.0
    for lg in flog:
        if "readout" not in lg:
            continue
        n_ro += 1
        tv = lg["tv"]
        lab = TL.readout_clock(tv)
        exp_clock = lab[1] if lab else None
        if exp_clock is not None:
            clock_err = max(clock_err, abs(exp_clock - lg["clock"]))
        s = fmt_mp(gap_deg_mp(lg["clock"]))
        if s != lg["readout"]:
            mism.append((lg["frame"], lg["readout"], s))
    R["readout_frames"] = n_ro
    R["readout_mismatches"] = mism[:10]
    R["readout_clock_max_err"] = clock_err
    by_frame = {lg["frame"]: lg for lg in flog}
    spot = {}
    for t in (1.0, 3.0, 5.0):
        exact = gap_deg_mp(t)
        fi = int(round((TL.REPLAY_A + t / TL.REPLAY_RATE) * 60))
        fi = min(fi, int(round(TL.REPLAY_B * 60)))
        spot[str(t)] = dict(formula_deg=mp.nstr(exact, 8), frame=fi, shown=by_frame[fi].get("readout"),
                            shown_clock=by_frame[fi].get("clock"))
    last = max(lg["frame"] for lg in flog if "readout" in lg)
    spot["last_readout_frame"] = dict(frame=last, clock=by_frame[last]["clock"], shown=by_frame[last]["readout"],
                                      formula_deg=mp.nstr(gap_deg_mp(by_frame[last]["clock"]), 8))
    R["gap_spot_checks"] = spot

    # ---------------- text: safe area + size, caption rules
    bad, min_size, n_txt = [], 1e9, 0
    for lg in flog:
        for tx in lg.get("text", []):
            if tx["alpha"] < 0.02:
                continue
            n_txt += 1
            min_size = min(min_size, tx["size"])
            x0, y0, x1, y1 = tx["box"]
            if x0 < SAFE["x0"] or x1 > SAFE["x1"] or y0 < SAFE["y0"] or y1 > SAFE["y1"]:
                bad.append((lg["frame"], tx["text"], tx["box"]))
    R["text_items_checked"] = n_txt
    R["text_outside_safe_area"] = bad[:10]
    R["text_min_font_px"] = min_size
    R["hook_words"] = TL.word_count(HOOK)
    R["captions"] = [dict(text=c.replace("\n", " "), words=TL.word_count(c), on_s=b - a) for c, a, b in TL.CAPTIONS]
    R["captions_ok"] = all(c["words"] <= 8 and c["on_s"] >= 1.5 for c in R["captions"]) and R["hook_words"] <= 6

    # ---------------- first vs last frame
    fl = decode(MASTER, [0, NFRAMES - 1])
    R["master_first_last_identical"] = bool(np.array_equal(fl[0], fl[NFRAMES - 1]))
    if os.path.exists(MAIN):
        fm = decode(MAIN, [0, NFRAMES - 1])
        d = fm[0].astype(float) - fm[NFRAMES - 1].astype(float)
        mse = np.mean(d ** 2)
        R["mp4_first_last_psnr_db"] = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
        R["mp4_first_last_max_abs_diff"] = int(np.max(np.abs(d)))

    # ---------------- stillness: longest run of identical consecutive frames (lossless master, loop-wrapped)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", MASTER, "-vf", "scale=360:640:flags=area", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    G = np.frombuffer(raw, np.uint8).reshape(-1, 640, 360).astype(np.int16)
    R["master_frames"] = int(G.shape[0])
    diff = np.array([np.abs(G[(i + 1) % len(G)] - G[i]).max() for i in range(len(G))])
    still = diff == 0
    runs, cur, best, best_at = [], 0, 0, None
    for i in range(2 * len(still)):
        if still[i % len(still)]:
            cur += 1
            if cur > best:
                best, best_at = cur, (i - cur + 1) % len(still)
        else:
            cur = 0
        if cur > len(still):
            break
    R["longest_still_run_frames"] = int(best + 1 if best else 1)
    R["longest_still_run_s"] = (best + 1) / 60 if best else 1 / 60
    R["longest_still_run_starts_at_frame"] = best_at
    # 'something new every 3 s': mean abs frame change per 3 s window is never ~0
    act = np.array([np.abs(G[min(i + 1, len(G) - 1)] - G[i]).mean() for i in range(len(G))])
    R["min_activity_per_3s_window"] = float(min(act[i:i + 180].max() for i in range(0, len(G) - 180, 30)))

    # ---------------- specs + loudness
    for name, path in (("main", MAIN), ("instagram", IG)):
        if not os.path.exists(path):
            continue
        pr = json.loads(run("ffprobe", "-v", "error", "-count_packets", "-show_entries",
                            "stream=codec_type,codec_name,profile,width,height,r_frame_rate,nb_read_packets,pix_fmt,"
                            "bit_rate,sample_rate,channels:format=duration,bit_rate", "-of", "json", path).stdout)
        v = next(s for s in pr["streams"] if s["codec_type"] == "video")
        a = next(s for s in pr["streams"] if s["codec_type"] == "audio")
        e = run("ffmpeg", "-nostats", "-i", path, "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-").stderr
        tail = e[e.rfind("Summary:"):]
        I = float(tail.split("I:")[1].split("LUFS")[0])
        TP = float(tail.split("Peak:")[1].split("dBFS")[0])
        ln = run("ffmpeg", "-nostats", "-i", path, "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1:print_format=json",
                 "-f", "null", "-").stderr
        lj = json.loads(ln[ln.rfind("{"):ln.rfind("}") + 1])
        mv = run("ffprobe", "-v", "error", "-show_entries", "format_tags=major_brand", "-of", "json", path)
        with open(path, "rb") as fh:
            head = fh.read(4096)
        R[name] = dict(width=v["width"], height=v["height"], fps=v["r_frame_rate"], frames=int(v["nb_read_packets"]),
                       duration_s=float(pr["format"]["duration"]), video_codec=v["codec_name"], profile=v["profile"],
                       pix_fmt=v["pix_fmt"], video_mbps=round(int(v["bit_rate"]) / 1e6, 2),
                       audio=f'{a["codec_name"]} {a["sample_rate"]} Hz {a["channels"]}ch {round(int(a["bit_rate"]) / 1e3)} kbps',
                       integrated_lufs_ebur128=I, true_peak_dbtp_ebur128=TP,
                       loudnorm_input_i=float(lj["input_i"]), loudnorm_input_tp=float(lj["input_tp"]),
                       faststart_moov_first=head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")),
                       size_mb=round(os.path.getsize(path) / 1e6, 1))

    # ---------------- stills at key moments (from the lossless master) + contact sheet
    os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
    ims = decode(MASTER, list(KEY.values()))
    paths = []
    for name, fi in KEY.items():
        p = os.path.join(HERE, "stills", f"key_{fi:04d}_{name}.png")
        Image.fromarray(ims[fi]).save(p)
        paths.append(p)
    subprocess.run([sys.executable, os.path.join(HERE, "sheet.py"), os.path.join(HERE, "stills", "key_moments_sheet.jpg")]
                   + paths + ["--w", "360"])
    R["stills"] = {k: f"stills/key_{v:04d}_{k}.png" for k, v in KEY.items()}

    json.dump(R, open(os.path.join(HERE, "data", "qa.json"), "w"), indent=2, default=str)
    write_log(meas, R)
    print(json.dumps(R, indent=1, default=str)[:6000])


def write_log(meas, R):
    L = ["# SimTheory #05 · Measurement log", "",
         "Generated by `sim.py` (physics) and `qa.py` (checks). Raw data: `data/measurements.json`, `data/qa.json`.", "",
         "## Simulation setup", "",
         f"- Simple pendulum: L = 1 m, m = 1 kg, g = 9.81 m/s², no friction; θ'' = −(g/L)·sin θ.",
         f"- Integrator: classic RK4, float64, dt = 1/4800 s = {meas['dt_s']*1e3:.4f} ms (≤ 0.5 ms), {meas['steps']} steps (0–40 s).",
         f"- ω0 = √(g/L) = {meas['omega0']:.6f} rad/s; critical bottom speed ω_c = 2ω0 = {meas['omega_c']:.6f} rad/s; E_c = 2mgL = {meas['E_c_J']:.2f} J.",
         "- Threshold pendulum (1.00): exact analytic solution θ = π − δ, δ(t) = 4·arctan(e^(−ω0·t)), gap computed directly from δ.",
         "", "## Per-pendulum results", "",
         "| E/E_c | push speed (rad/s) | method | max relative energy drift | phase-curve max error | flips over top (0–40 s) |",
         "|---|---|---|---|---|---|"]
    for p in meas["pendulums"]:
        L.append(f"| {p['fraction']:.2f} | {p['push_speed_rad_s']:.4f} | {p['method']} | {p['energy_rel_drift_max']:.2e} | "
                 f"{p['phase_curve_max_err']:.2e} | {p.get('flips_over_top', 'never arrives')} |")
    L += ["", "Energy-drift limit is 1e-4 (0.01 %); phase-curve limit is 5e-3 (0.5 %). Phase-curve error = max over all "
          "steps of | |ω_sim| − √(2E/(mL²) − 2ω0²(1 − cos θ)) |, divided by the pendulum's push speed.", "",
          "## Threshold checks", "",
          f"- 0.95 never flipped: **{meas['check_095_never_flipped']}** (max angle {meas['check_095_max_angle_deg']:.4f}°; exact amplitude "
          f"{[p for p in meas['pendulums'] if p['fraction'] == 0.95][0]['theta_max_exact_deg']:.4f}°).",
          f"- 1.05 always flipped: **{meas['check_105_always_flipped']}** (minimum spin speed {meas['check_105_min_omega']:.4f} rad/s > 0; "
          f"{meas['check_105_flips_in_40s']} flips in 40 s).",
          f"- Analytic solution: ODE residual max {meas['analytic_ode_residual_max']:.1e} rad/s², energy error max {meas['analytic_energy_rel_err_max']:.1e}.",
          f"- For contrast, a numerically integrated threshold pendulum leaves the top at t ≈ {meas['numeric_threshold_leaves_top_at_s']:.2f} s "
          f"({meas['numeric_threshold_fate']}): rounding knocks it off the knife-edge, so the video draws it from the exact formula.",
          "", "## Gap to the top, δ (degrees, computed directly from δ(t))", "",
          "| t since push | formula (60-digit check) | shown on screen | frame |", "|---|---|---|---|"]
    for k, v in R["gap_spot_checks"].items():
        lab = f"{k} s" if k != "last_readout_frame" else f"{v['clock']:.2f} s (last readout frame)"
        L.append(f"| {lab} | {v['formula_deg']}° | {v['shown']} | {v['frame']} (clock {v['shown_clock'] if 'shown_clock' in v else v['clock']:.4f} s) |")
    L += ["", "The replay clock runs 0 → 5 s over video 28–38 s (0.5×); frame 2280 (video 38.000 s) shows t = 5.00 s. "
          "From the next frame the readout runs on live time since the push. The final video frame (2519) is the loop pose "
          "(all pendulums at rest, readout hidden); the last frame with a readout is listed above.", "",
          "## On-screen numbers vs log", "",
          f"- Gap readout frames checked: {R['readout_frames']}; mismatches vs independent 60-digit recomputation: {len(R['readout_mismatches'])}.",
          f"- Replay / live clock max error vs the time map: {R['readout_clock_max_err']:.1e} s.",
          f"- Text items checked: {R['text_items_checked']}; outside the safe area: {len(R['text_outside_safe_area'])}; smallest font: {R['text_min_font_px']} px.",
          f"- Captions ≤ 8 words and ≥ 1.5 s, hook ≤ 6 words: {R['captions_ok']}.",
          "", "## Video / audio", ""]
    for name in ("main", "instagram"):
        if name in R:
            v = R[name]
            L.append(f"- {name}: {v['width']}×{v['height']}, {v['fps']} fps, {v['frames']} frames, {v['duration_s']:.3f} s, "
                     f"{v['video_codec']} {v['profile']} {v['pix_fmt']} {v['video_mbps']} Mbps, {v['audio']}; "
                     f"{v['integrated_lufs_ebur128']} LUFS, true peak {v['true_peak_dbtp_ebur128']} dBTP (loudnorm: "
                     f"{v['loudnorm_input_i']} LUFS / {v['loudnorm_input_tp']} dBTP); faststart: {v['faststart_moov_first']}; {v['size_mb']} MB.")
    L += [f"- First vs last frame: lossless master identical = {R['master_first_last_identical']}; "
          f"MP4 PSNR = {R.get('mp4_first_last_psnr_db')} dB, max abs diff = {R.get('mp4_first_last_max_abs_diff')}.",
          f"- Longest completely still stretch (loop-wrapped): {R['longest_still_run_frames']} frames = {R['longest_still_run_s']:.2f} s "
          f"(the hook hold across the loop seam).", ""]
    open(os.path.join(HERE, "measurement_log.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
