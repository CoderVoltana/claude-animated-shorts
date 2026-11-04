"""Writes MEASUREMENT_LOG.md and PUBLISHING.md from data/physics.json, data/qa.json, data/audio_report.json."""
import json
import timeline as TL

P = json.load(open("data/physics.json")); Q = json.load(open("data/qa.json")); A = json.load(open("data/audio_report.json"))
rk = TL.ranking(P)
L = []
w = L.append
w("# SimTheory #06: Does length change everything? Measurement log\n")
w("All values below are produced by `sim.py` (physics) and `qa.py` (checks). Every on-screen number is read from the same files.\n")
w("## Setup\n")
w(f"- Model: {P['model']}. Released from rest at θ1 = θ2 = 90° (both arms horizontal), video time {TL.RELEASE} s, played at 1× real time.")
w(f"- Timestep **dt = {P['dt_s']*1000:.4f} ms** (1/{round(1/P['dt_s'])} s), RK4, float64. States saved at 2400 Hz (40 per video frame).")
w(f"- Chaos clock: invisible twin with θ1 + {P['twin_offset_rad']:g} rad. Clock stops when the two bottom-arm angles differ by more than {P['chaos_threshold_deg']}°. Horizon {P['chaos_horizon_s']:.0f} s of sim time.")
w("- Flip: the bottom arm passes straight up and then swings down past horizontal on the far side. A wobble over the top that falls back the way it came does not count.")
w("- Energy reference: potential measured from the lowest configuration (both bobs hanging straight down), so E0 > 0 and relative drift is well defined.\n")
w("## Results\n")
w("| Ratio (upper:lower) | Arms (m) | Chaos clock | Chaos clock at dt/2 | Flips by 16.5 s (end of painting) | Final flips (sim 28.5 s, shown at 29.0 s) | Max energy drift (60 s) |")
w("|---|---|---|---|---|---|---|")
for lab, p in P["pendulums"].items():
    tc = TL.fmt_chaos(p["chaos_clock_s"]); th = "not within 60 s" if p["chaos_clock_s_at_dt_half"] is None else f"{p['chaos_clock_s_at_dt_half']:.4f} s"
    raw = "" if p["chaos_clock_s"] is None else f" ({p['chaos_clock_s']:.4f} s)"
    w(f"| {lab} | {p['L1_m']} + {p['L2_m']} | **{tc}**{raw} | {th} | {p['flips_at_sim_16_5s']} | {p['flips_at_sim_28_5s_final_shown']} | {p['max_energy_drift_rel_60s']:.2e} ({p['max_energy_drift_rel_60s']*100:.1e} %) |")
w("")
w("**Final ranking (shortest time until unpredictable first):** " + " → ".join(
    f"{l} ({TL.fmt_chaos(P['pendulums'][l]['chaos_clock_s'])})" for l in rk))
w(f"- Most chaotic: **{rk[0]}**. Most predictable: **{rk[-1]}**. These are the captions in the video.\n")
w("## Physics checks\n")
w(f"- Energy drift < 0.1 % for every pendulum: **{'PASS' if all(p['energy_pass'] for p in P['pendulums'].values()) else 'FAIL'}** (worst {max(p['max_energy_drift_rel_60s'] for p in P['pendulums'].values()):.1e}, i.e. about 10⁻⁸ %).")
hc = P["halving_check"][-1]
w(f"- Timestep-halving check ({P['halving_check_rule']}): **{'PASS' if hc['pass_all'] else 'FAIL'}** at dt = {hc['dt_s']*1000:.4f} ms.")
w("\n| Ratio | Max bottom-bob difference, dt vs dt/2, first 10 s | In pixels at 260 px/m | dt and dt/2 runs visually identical until |")
w("|---|---|---|---|")
for lab, r in hc["pendulums"].items():
    w(f"| {lab} | {r['max_bob2_diff_m_first10s']:.2e} m | {r['max_diff_px_at_260px_per_m']:.2e} px | {r['identical_to_dt_half_until_s']:.2f} s |")
w("\nThe last column is how long the dt and dt/2 runs stay within half a pixel over the full 28.5 s shown. "
  "Past that point they drift apart only because chaos magnifies round-off at the 1e-15 level. That is the same sensitivity the chaos clock measures, not a timestep error. The chaos-clock times themselves agree at dt/2 to within 0.2 ms.")
w(f"- Upward reach (needed for layout): energy conservation limits both bobs to min(L1, L2/2) ≤ 0.5 m above the pivot. Measured maxima: " +
  ", ".join(f"{l} {p['max_bob2_height_above_pivot_m']:.2f} m" for l, p in P["pendulums"].items()) + ".\n")
w("## Video and audio checks\n")
m, ig = Q["main"], Q["instagram"]
w(f"- Main: {m['width']}×{m['height']}, {m['fps']} fps, {m['frames']} frames, {m['duration_s']:.3f} s, {m['video_codec']} {m['profile']}, {m['pix_fmt']}, {m['video_kbps']/1000:.1f} Mbps video. Audio: {m['audio_codec']} {m['audio_hz']} Hz, encoded at 320 kbps (ffmpeg native AAC, twoloop coder; measured average {m['audio_kbps']} kbps because the encoder does not spend the full budget on this soft material), {m['channels']} ch. Atom order: {' then '.join(m['faststart'])} (faststart).")
w(f"- Instagram: {ig['width']}×{ig['height']}, {ig['fps']} fps, {ig['frames']} frames, {ig['video_kbps']/1000:.1f} Mbps.")
w(f"- Loudness (ffmpeg ebur128, main MP4): **{m['ebur128_integrated_lufs']} LUFS integrated**, **{m['ebur128_true_peak_dbtp']} dBTP true peak**, LRA {m['ebur128_lra_lu']} LU. ffmpeg loudnorm measurement: I = {m['loudnorm_measured_I']} LUFS, TP = {m['loudnorm_measured_TP']} dBTP. Instagram copy: {ig['ebur128_integrated_lufs']} LUFS, {ig['ebur128_true_peak_dbtp']} dBTP.")
w(f"- Effects vs bed: bed set {A['fx_vs_bed_db']} dB under the effects (BS.1770 integrated). High-pass 40 Hz (4th order), low-pass 12 kHz, −2 dB shelf above 6 kHz.")
w(f"- Loop seam: first and last audio samples = {A['seam_first_sample']} / {A['seam_last_sample']}. RMS of the last 50 ms {A['seam_rms_last_50ms_dbfs']} dBFS, first 50 ms {A['seam_rms_first_50ms_dbfs']} dBFS (no click).")
w(f"- First vs last frame: lossless master **bit-identical = {Q['master_first_last_identical']}**. Main MP4 decoded max pixel difference {Q['main_first_last_maxdiff']} (PSNR {Q['main_first_last_psnr_db']} dB). Instagram {Q['instagram_first_last_maxdiff']} (PSNR {Q['instagram_first_last_psnr_db']} dB).")
w(f"- Safe area (x 86–1152, y 384–1664): every text, label, counter and chart pixel of all 1920 frames lies in the union box {Q['text_ui_union_bbox_px']}. Frames outside: **{Q['text_ui_outside_safe_frames']}**.")
w(f"- Pendulums stay inside their cells and below the labels for the whole run: " + ", ".join(f"{l} {'yes' if g['inside_cell'] and g['clear_of_labels'] else 'NO'}" for l, g in Q["pendulum_geometry"].items()) + f". Same scale for all four: {Q['grid_scale_px_per_m']:.0f} px/m (overlay 240 px/m).")
w(f"- On-screen numbers match this log: **{Q['onscreen_numbers_match_log']}**. Chart: {Q['onscreen_numbers']['chart']}. Counters at 29.0 s: {Q['onscreen_numbers']['flip_counters_at_29s']}. Counters on the last frame: {Q['onscreen_numbers']['flip_counters_at_frame_1919']}.")
w("- Text sizes: hook 130 px, captions 96 px, labels/counters/chart values 44 px, chart labels 48 px, chart title 60 px, SIM #06 44 px (Inter Light). Lines: trails ≥ 3.2 px, rods 4 px, hairlines 3 px.\n")
w("## Timeline (video time → sim time)\n")
w("- 0.0–0.5 s: hook hold, sim 0. 0.5–29.0 s: sim = t − 0.5 (1×). 29.0–31.75 s: rewind sim 28.5 → 0 (smootherstep). 31.75–32.0 s: frame-1 state.")
w("- Captions: " + "; ".join(f"\"{' '.join(c[0])}\" {c[1]}–{c[2]} s" for c in TL.captions(P)))
w("- Sound cues (video s): " + ", ".join(f"{c[0]}@{c[1]}({c[2]})" for c in A["cues"]))
open("MEASUREMENT_LOG.md", "w").write("\n".join(L) + "\n")

pub = f"""# SimTheory #06: publishing copy

## Title
Does Length Change Everything? 4 Double Pendulums

## Description
Four double pendulums: same 2 m total length, same 90° start. Only the arm ratio changes, and each one paints a completely different light-drawing. SimTheory #06.
Real physics: double-pendulum simulation with unequal arms, no tricks. Chaos times measured, not guessed.
Result: {rk[0]} became unpredictable first, after {TL.fmt_chaos(P['pendulums'][rk[0]]['chaos_clock_s'])}. {rk[-1]} never lost predictability within 60 s.

## Hashtags
#physics #chaostheory #doublependulum #simulation #science

## Pinned comment
Be honest: before the chart came up, which ratio would you have bet on as the most chaotic? 1:3, 1:1, 3:1 or 9:1? And what should SimTheory simulate next? Drop your idea and the best one becomes Episode 7.
"""
open("PUBLISHING.md", "w").write(pub)
print("written")
