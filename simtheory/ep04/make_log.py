"""Writes out/MEASUREMENT_LOG.md from data/measure.json, data/audio_report.json and out/qa_report.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")
M = json.load(open(os.path.join(HERE, "data", "measure.json")))
A = json.load(open(os.path.join(HERE, "data", "audio_report.json")))
Q = json.load(open(os.path.join(OUT, "qa_report.json")))
sm = Q["checks"]["stream_specs_main"]["detail"]; si = Q["checks"]["stream_specs_instagram"]["detail"]
lm, li = Q["loudness_main"], Q["loudness_instagram"]
f = lambda x, n=2: f"{x:.{n}f}"
rows = [
    ("Damping (viscous) c", f"{M['c_viscous_Nms']} N·m·s/rad at both joints (joint 1: −c·ω1, joint 2: −c·(ω2−ω1))"),
    ("Dry joint friction μ (added, see note)", f"{M['mu_dry_Nm']} N·m, smoothed as μ·tanh(ω_rel/ω0), ω0 = {M['w0_rad_s']} rad/s"),
    ("Twin offset (θ1, both lanes)", f"{M['twin_offset_deg']}°"),
    ("Start / release", f"θ1 = θ2 = {M['theta0_deg']:.0f}° at rest, released at video 0.5 s"),
    ("Integrator", f"{M['integrator']}, dt = {M['dt_s']*1000:.4f} ms (1/1200 s)"),
    ("Playback speed", f"1× from 0.5 s to 10.5 s (whole chaotic phase), smooth ramp to {M['rmax_timelapse']:.2f}× (10.5–14.5 s), "
                       f"back to 1× (23.5–25.5 s), 1× to 31 s; labelled in the HUD at all times. Rewind 31–33.85 s."),
    ("Sim time shown", f"0 → {M['sim_end_s']:.1f} s"),
    ("Friction twins visibly split (>5° / >20°)", f"{f(M['split_vis_right_video'])} s / {f(M['split20_right_video'])} s video"),
    ("No-friction twins visibly split (>5° / >20°)", f"{f(M['split_vis_left_video'])} s / {f(M['split20_left_video'])} s video"),
    ("Friction lane stopped flipping (last arm over the top)", f"{f(M['last_flip_right_video'])} s video (sim {f(M['last_flip_right_sim'])} s); flips per twin {M['flips_right']}"),
    ("Re-merge (< 0.5° apart, both arms, for good)", f"{f(M['remerge_right_video'])} s video (sim {f(M['remerge_right_sim'])} s)"),
    ("Rest (all speeds < 0.5°/s, for good)", f"{f(M['rest_right_video'])} s video (sim {f(M['rest_right_sim'])} s)"),
    ("Final twin separation, friction lane", f"{M['final_sep_right_deg']:.2e}° (angles within {max(abs(x) for x in M['final_right_angles_deg']):.1e}° of straight down)"),
    ("Final twin separation, no-friction lane", f"{M['final_sep_left_deg']:.1f}° (bottom arm {M['final_sep_left_bottom_deg']:.1f}°)"),
    ("Final energy left, no-friction lane", f"{M['final_energy_left_pct'][0]:.7f}% / {M['final_energy_left_pct'][1]:.7f}% (shown: 100%)"),
    ("Final energy left, friction lane", f"{M['final_energy_right_pct'][0]:.1e}% / {M['final_energy_right_pct'][1]:.1e}% (shown: 0%)"),
    ("Max energy drift, no-friction lane", f"{M['max_energy_drift_left_rel_E0']*100:.2e}% of E0 ({M['max_energy_drift_left_rel_available']*100:.2e}% of available energy) — limit 0.1%"),
    ("Max energy rise per step, friction lane", f"{M['max_energy_rise_right_rel_per_step']:.1e} (relative) — limit 1e-9"),
    ("Flips, no-friction lane (per twin)", f"{M['flips_left']}"),
]
L = ["# SimTheory #04 — measurement log", "",
     "Energy % = (E − E_rest) / (E0 − E_rest), with E_rest = −3mgl (hanging straight down) and E0 = energy at release "
     f"({M['E0_J']:.3f} J; available {M['E_available_J']:.3f} J). Angle separations are wrapped to ±180° (mod 360°).", "",
     "| Quantity | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
L += ["", "## Physics checks", ""] + [f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in M["checks"].items()]
L += ["", "## Delivery checks", ""] + [f"- {'PASS' if v['pass'] else 'FAIL'} — {k}" for k, v in Q["checks"].items()]
L += ["", "## Video / audio", "",
      f"- Main: {sm['res']}, {sm['fps']} fps, {sm['frames']} frames, {sm['duration_s']:.3f} s, H.264 {sm['profile']} {sm['pix_fmt']}, "
      f"{sm['video_mbps']} Mbps video, AAC {sm['audio_kbps']} kbps {sm['audio_sr']} Hz, {sm['size_mb']} MB",
      f"- Instagram: {si['res']}, {si['fps']} fps, {si['frames']} frames, {si['video_mbps']} Mbps video, {si['size_mb']} MB",
      f"- Loudness (ffmpeg ebur128, main MP4): integrated {lm['ebur128_integrated_lufs']} LUFS, true peak {lm['ebur128_true_peak_dbtp']} dBTP; "
      f"loudnorm measure: I {lm['loudnorm_input_i']} LUFS, TP {lm['loudnorm_input_tp']} dBTP",
      f"- Loudness (Instagram MP4): integrated {li['ebur128_integrated_lufs']} LUFS, true peak {li['ebur128_true_peak_dbtp']} dBTP",
      f"- Mix master (WAV, before AAC): {A['integrated_lufs']:.2f} LUFS, {A['true_peak_dbtp']:.2f} dBTP; loop edges |x| ≤ {max(A['start_abs_max'], A['end_abs_max']):.1e}",
      "- Sound cues (video s): " + ", ".join(f"{c['cue']} {c['video_t']}" for c in A["cues"]),
      f"- Energy readouts shown (28–31 s): {Q['readouts_shown']}",
      "", "## Why there is dry friction and a labelled time-lapse (honesty note)", "",
      "With the brief's friction law alone (linear viscous, −c·ω at each joint), the timing targets cannot be met at any single "
      "playback speed. Viscous joint damping acts on how fast each joint turns. Once the chaos is over, both arms swing almost "
      "together, so the remaining swing loses energy very slowly (amplitude decay ≈ 0.086·c per second). "
      "Measured: c = 0.05 keeps chaos to ~10 s, but the pendulum still holds ~9% of its energy after 30 s and needs "
      "well over 1000 s to come to rest. c ≥ 0.1 removes the chaos entirely (the twins never split at a 120° start). "
      "The time from 'chaos ends' to 'at rest' is more than 100× the chaos time, and a constant speed scales both equally.",
      "",
      "Fix (still real physics, only the parameters were tuned, never the visuals): the viscous term is kept exactly as briefed, and each joint also gets "
      "ordinary dry (Coulomb) bearing friction μ·tanh(ω/ω0), the same at both joints and for both twins. It always opposes the motion, so energy "
      "can only fall (checked every step). Even so, settling takes ~70 s of sim time after the chaos. So playback stays 1× for the whole chaotic "
      "phase, then a smooth, always-labelled time-lapse (up to "
      f"{M['rmax_timelapse']:.1f}×) carries the settling, and playback returns to 1× for the final comparison. "
      "The sim clock (t = … s) and speed are on screen in every frame.",
      ]
open(os.path.join(OUT, "MEASUREMENT_LOG.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
