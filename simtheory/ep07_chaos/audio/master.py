"""Stage 5: balance stems, filter for phones, normalise to -14 LUFS / <= -1 dBTP, verify with ffmpeg ebur128."""
import json
import os
import re
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "audio")
SR = 48000
BED_UNDER_FX_DB = 9.0
TARGET_LUFS = -14.0
TP_MAX = -1.0


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true:framelog=quiet",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    summ = r[r.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summ).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summ).group(1))
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summ).group(1))
    return I, tp, lra


def write(path, x):
    wavfile.write(path, SR, x.astype(np.float32))


def db(x):
    return 10 ** (x / 20)


bed = wavfile.read(os.path.join(OUT, "bed.wav"))[1].astype(np.float64)
fx = wavfile.read(os.path.join(OUT, "fx.wav"))[1].astype(np.float64)
I_bed, _, _ = ebur128(os.path.join(OUT, "bed.wav"))
I_fx, _, _ = ebur128(os.path.join(OUT, "fx.wav"))
bed_gain_db = (I_fx - BED_UNDER_FX_DB) - I_bed
mix = fx + bed * db(bed_gain_db)

# phone-friendly: high-pass below ~40 Hz, gentle low-pass against harsh highs
mix = sosfiltfilt(butter(2, 40, "highpass", fs=SR, output="sos"), mix, axis=0)
mix = sosfiltfilt(butter(2, 11000, "lowpass", fs=SR, output="sos"), mix, axis=0)
# keep the loop seam silent (filters can smear a few samples)
ramp = np.minimum(1, np.minimum(np.arange(len(mix)), np.arange(len(mix))[::-1]) / 48)
mix *= ramp[:, None]

tmp = os.path.join(OUT, "_tmp.wav")
gain = 0.0
for it in range(6):
    write(tmp, mix * db(gain))
    I, tp, lra = ebur128(tmp)
    if abs(I - TARGET_LUFS) < 0.1 and tp <= TP_MAX:
        break
    gain += TARGET_LUFS - I
    # if true peak would exceed the ceiling, soft-limit the peaks (lookahead-free smooth knee, 4x oversampled check by ebur128)
    if tp + (TARGET_LUFS - I) > TP_MAX:
        x = mix * db(gain)
        ceil = db(TP_MAX - 0.4)
        mix = np.where(np.abs(x) > 0.5 * ceil, np.sign(x) * (0.5 * ceil + 0.5 * ceil * np.tanh((np.abs(x) - 0.5 * ceil) / (0.5 * ceil))), x) / db(gain)
final = mix * db(gain)
write(os.path.join(OUT, "mix.wav"), final)
I, tp, lra = ebur128(os.path.join(OUT, "mix.wav"))
os.remove(tmp)
rep = {"bed_lufs_raw": I_bed, "fx_lufs_raw": I_fx, "bed_gain_db": bed_gain_db, "bed_under_fx_db": BED_UNDER_FX_DB,
       "integrated_lufs": I, "true_peak_dbtp": tp, "lra_lu": lra, "samples": len(final),
       "first_sample": float(np.abs(final[0]).max()), "last_sample": float(np.abs(final[-1]).max())}
json.dump(rep, open(os.path.join(OUT, "loudness.json"), "w"), indent=2)
print(json.dumps(rep, indent=2))
