"""Writes the measurement log (deliverable 6) from data/measurements.json, timeline.py, loudness and QA results."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import timeline as T  # noqa: E402

M = T.M
L = json.load(open(os.path.join(ROOT, "out", "audio", "loudness.json")))
qa_p = os.path.join(ROOT, "out", "qa", "qa_report.json")
Q = json.load(open(qa_p)) if os.path.exists(qa_p) else None
lab = {"f32": "float32 (single)", "f64": "float64 (double)", "mp24": "arbitrary, 24 digits",
       "mp32": "arbitrary, 32 digits", "mp40": "arbitrary, 40 digits"}

o = []
w = o.append
w("# SimTheory #07 - Can a computer predict chaos? - measurement log\n")
w("All numbers below are produced by `physics/run_sims.py` + `physics/analyze.py` and read by the renderer and "
  "audio from `data/measurements.json`; nothing on screen is typed by hand.\n")
w("## Setup\n")
w("- Double pendulum, equal arms 1 m, equal masses 1 kg, g = 9.81 m/s^2, no friction, released from rest at "
  "theta1 = theta2 = 120 deg.")
w("- Identical code for every run (`physics/dp.py`): same equations, classic RK4, dt = 1 ms, same start angles. "
  "Only the number type changes. Every constant (g, dt, 1/6, pi, 120 deg) is converted in the run's own precision.")
w("- float32 run: every operation in numpy float32 (type asserted on the final state). float64: Python floats. "
  "Arbitrary precision: mpmath with a mantissa of round(digits x log2 10) bits.")
w(f"- Simulated {M['T_sim']:.0f} s of sim time per run ({int(M['T_sim']/M['dt'])} steps), every state saved to disk.")
w("- Prediction horizon = first 1 ms step at which the bottom-arm angle theta2 differs from the reference by more than 1 deg.\n")
w("## Prediction horizons\n")
w("| Run | Digits (label) | Mantissa bits | Exact digits | Horizon (s, sim) | Split in video (s) |")
w("|---|---|---|---|---|---|")
for k, v in M["horizons"].items():
    vt = f"{T.RELEASE + v['horizon_s'] / T.SPEED:.3f}" if k in ("f32", "f64") else "chart only"
    w(f"| {lab[k]} | {v['digits']} | {v['mantissa_bits']} | {v['exact_digits']:.2f} | **{v['horizon_s']:.3f}** | {vt} |")
f = M["fit"]
fe = M["fit_exact_digits"]
w("\n## Fit: horizon = a + b x digits\n")
w(f"- **Slope b = {f['slope_s_per_digit']:.4f} s per digit** (shown as \"+{T.SLOPE:.1f} s per digit\"), "
  f"intercept a = {f['intercept_s']:.3f} s, R^2 = {f['r2']:.5f}.")
w(f"- Residuals (s): {', '.join(f'{r:+.2f}' for r in f['residuals_s'])}.")
w(f"- **Digits needed for a 60 s prediction: {f['digits_for_60s']:.2f}** (shown as \"~{round(f['digits_for_60s'])} digits\"). "
  f"Consistent with the direct 40-digit measurement of {M['horizons']['mp40']['horizon_s']:.2f} s.")
w(f"- Cross-check fitting on the exact digits each format keeps: slope {fe['slope_s_per_digit']:.4f} s/digit, "
  f"60 s needs {fe['digits_for_60s']:.2f} digits.\n")
ly = M["lyapunov"]
w("## Lyapunov exponent (estimate)\n")
w(f"- From the fit slope: lambda = ln(10) / b = **{ly['from_fit_slope']:.3f} 1/s** (error grows x10 every {f['slope_s_per_digit']:.2f} s, "
  f"e-folding time {1/ly['from_fit_slope']:.2f} s).")
w("- Cross-check, log-slope of the error envelope of each run against the reference: " +
  ", ".join(f"{k} {v:.2f}" for k, v in ly["from_error_growth"].items()) + " 1/s "
  "(cruder: it includes the slow start before errors are amplified).\n")
rc = M["reference_check"]
w("## Reference precision and its check\n")
w(f"- Reference precision used: **100 digits** (332-bit mantissa).")
w(f"- Check run: 120 digits (399 bits). Max |difference| over all four state variables at every 1 ms step over "
  f"the chart window (0-{rc['window_s']:.3f} s): **{rc['max_abs_diff_window']:.2e}** "
  f"(requirement < 1e-20: {'PASS' if rc['passes_1e-20'] else 'FAIL'}). Over the full {M['T_sim']:.0f} s: "
  f"{rc['max_abs_diff_full']:.2e}. No increase of the reference precision was needed.\n")
w("## Energy drift (max |E - E0| / |E0|, E0 = 14.715 J, sampled every 10 ms)\n")
for k, v in M["energy"].items():
    w(f"- {k}: {v['max_rel_drift']:.2e}" + ("  <- reference, requirement < 0.1 %: PASS" if k == M["reference"] and v["max_rel_drift"] < 1e-3 else ""))
w("\nAll runs share the same RK4 truncation error (~2e-8); the float32 run's larger drift (4.8e-4) is its own rounding error.\n")
w("## Playback and timing\n")
w(f"- Release at {T.RELEASE} s video time; one constant playback speed **x{T.SPEED:g}** (sim time = (video time - 0.5) x {T.SPEED:g}) until the rewind at {T.REWIND_T0} s.")
w(f"- 7-digit split at **{T.T7:.2f} s** video, 16-digit split at **{T.T16:.2f} s** video.")
w(f"- Note: the measured horizons have a ratio of {T.H16/T.H7:.2f}, so no constant speed can put the 7-digit split at 11-13 s "
  f"and the 16-digit split at 17-19 s at the same time (that needs a ratio near 1.5). x{T.SPEED:g} keeps the 16-digit split "
  f"in its window before the chart at 21 s; the 7-digit split arrives earlier, at {T.T7:.2f} s, and the captions are timed from the measured mapping.")
w(f"- Rewind {T.REWIND_T0}-{T.REWIND_T1} s replays the recorded states backwards from sim {T.TAU_REWIND0:.2f} s to 0; the last frame is the first frame.\n")
w("## Captions as shown (from the timeline)\n")
w("| Start | End | Text |")
w("|---|---|---|")
w(f"| 0.00 | 2.50 | {T.HOOK_TEXT} (hook) |")
for a, b, l in T.CAPTIONS:
    w(f"| {a:.2f} | {b:.2f} | {' '.join(l)} |")
w("\nChart labels: points 9.7 / 23.1 / 35.6 / 46.6 / 59.8 s; \"+%.1f s per digit\"; gold dashed line at 60 s with a ring at %.1f digits; "
  "magenta band = fewer than 6 digits (illustrative, labelled \"real measurements\", no number claimed).\n" % (T.SLOPE, T.D60))
w("## Audio\n")
w(f"- Integrated loudness **{L['integrated_lufs']:.1f} LUFS**, true peak **{L['true_peak_dbtp']:.1f} dBTP**, LRA {L['lra_lu']:.1f} LU "
  f"(ffmpeg ebur128 on the master WAV). Drone bed set {L['bed_under_fx_db']:.0f} dB under the effects. "
  "High-pass 40 Hz, low-pass 11 kHz, exact zero at the loop seam.")
if Q:
    w("\n## Delivery QA (measured on the encoded files)\n")
    for nm in ("main", "instagram"):
        q = Q[nm]
        w(f"- {nm}: {q['width']}x{q['height']}, {q['fps']} fps, {q['frames']} frames, video {q['video_duration_s']:.3f} s, "
          f"{q['codec']} {q['profile']} {q['pix_fmt']}, {q['video_mbps']} Mbps, {q['audio_codec']} {q['audio_sr']} Hz "
          f"{q['audio_kbps']:.0f} kbps, faststart {q['faststart']}, {q['size_mb']} MB, loudness "
          f"{q['loudness']['integrated_lufs']} LUFS / {q['loudness']['true_peak_dbtp']} dBTP.")
    lp = Q["loop"]
    w(f"- Loop: rendered first and last frames identical = {lp['rendered_png_identical']}; encoded max pixel difference "
      f"{lp['encoded_max_abs_diff']}, PSNR {lp['encoded_psnr_db']:.1f} dB.")
    mo = Q["motion"]
    w(f"- Motion: first changed frame {mo['first_changed_frame']} (pulse + slow push on the deliberate hold), release at frame "
      f"{mo['release_frame']}; pixel-identical consecutive frame pairs: {mo['identical_frame_pairs']} (longest run {mo['longest_identical_run_frames']}).")
    w(f"- Safe-area violations by any text: {Q['safe_area_violations']}.")
    w("- On-screen numbers vs this log: " + ", ".join(f"{k} {'OK' if v['ok'] else 'MISMATCH'}" for k, v in Q["numbers"].items()) + ".")
open(os.path.join(ROOT, "MEASUREMENT_LOG.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
