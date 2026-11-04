"""Stage 2: measurements from the saved runs -> data/measurements.json"""
import os, sys, gzip, json, math
import numpy as np
import mpmath

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DT = 0.001
THR = math.radians(1.0)
DIGITS = {"f32": 7, "f64": 16, "mp24": 24, "mp32": 32, "mp40": 40}
REF = os.environ.get("REF", "mp100")
CHK = os.environ.get("CHK", "mp120")

load = lambda n: np.load(os.path.join(DATA, f"run_{n}.npy"))
ref = load(REF)
N = len(ref) - 1

res = {"dt": DT, "threshold_deg": 1.0, "T_sim": N * DT, "reference": REF, "check_run": CHK, "horizons": {}}
for name, d in DIGITS.items():
    r = load(name)
    diff = np.abs(r[:, 1] - ref[:, 1])
    idx = np.nonzero(diff > THR)[0]
    if len(idx) == 0:
        raise SystemExit(f"{name}: no horizon within {N*DT}s; extend T_SIM")
    i = int(idx[0])
    res["horizons"][name] = {"digits": d, "horizon_s": i * DT, "step": i}
    print(f"{name:5s} digits={d:3d} horizon={i*DT:7.3f} s")

# linear fit horizon vs digits
x = np.array([v["digits"] for v in res["horizons"].values()], float)
y = np.array([v["horizon_s"] for v in res["horizons"].values()], float)
b, a = np.polyfit(x, y, 1)
yhat = a + b * x
r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
d60 = (60 - a) / b
lam_fit = math.log(10) / b
res["fit"] = {"slope_s_per_digit": b, "intercept_s": a, "r2": r2, "digits_for_60s": d60,
              "residuals_s": (y - yhat).tolist()}
# cross-check: fit on the exact digits each format keeps (mantissa bits * log10 2)
BITS = {"f32": 24, "f64": 53, "mp24": 80, "mp32": 106, "mp40": 133}
xe = np.array([BITS[k] * math.log10(2) for k in res["horizons"]])
be, ae = np.polyfit(xe, y, 1)
for k in res["horizons"]:
    res["horizons"][k]["mantissa_bits"] = BITS[k]
    res["horizons"][k]["exact_digits"] = BITS[k] * math.log10(2)
res["fit_exact_digits"] = {"slope_s_per_digit": be, "intercept_s": ae, "digits_for_60s": (60 - ae) / be}
print(f"fit on exact digits: slope={be:.4f} s/digit, intercept={ae:.3f}, digits for 60 s={(60-ae)/be:.2f}")
print(f"fit: horizon = {a:.3f} + {b:.4f} * digits   R^2={r2:.5f}   digits for 60 s = {d60:.2f}")
print(f"Lyapunov from slope: ln10/slope = {lam_fit:.4f} 1/s")

# Lyapunov from error growth of each run vs reference (log|dtheta2| slope in the 1e-? .. 1e-3 band)
lam_growth = {}
for name in DIGITS:
    r = load(name)
    e = np.abs(r[:, 1] - ref[:, 1]) + np.abs(r[:, 0] - ref[:, 0])
    t = np.arange(len(e)) * DT
    h = res["horizons"][name]["step"]
    # running max envelope to smooth oscillation, fit over 10%..95% of horizon
    env = np.maximum.accumulate(e[:h])
    lo = int(0.15 * h)
    m = env[lo:] > 0
    if m.sum() > 50:
        sl, _ = np.polyfit(t[lo:h][m], np.log(env[lo:][m]), 1)
        lam_growth[name] = sl
print("Lyapunov from error-growth envelopes:", {k: round(v, 3) for k, v in lam_growth.items()})
res["lyapunov"] = {"from_fit_slope": lam_fit, "from_error_growth": lam_growth,
                   "growth_mean_mp": float(np.mean([lam_growth[k] for k in lam_growth if k.startswith("mp")]))}

# Energy drift (reference and others), energy sampled every 10 steps
res["energy"] = {}
for name in list(DIGITS) + [REF, CHK]:
    en = np.load(os.path.join(DATA, f"energy_{name}.npy"))
    drift = np.max(np.abs(en - en[0])) / abs(en[0])
    res["energy"][name] = {"E0": float(en[0]), "max_rel_drift": float(drift)}
print("energy drift:", {k: f"{v['max_rel_drift']:.2e}" for k, v in res["energy"].items()})

# Reference check: CHK vs REF at full precision at every step over the chart window
window_steps = max(v["step"] for v in res["horizons"].values())
mpmath.mp.dps = 200
maxd = mpmath.mpf(0); at = 0
with gzip.open(os.path.join(DATA, f"hp_{REF}.txt.gz"), "rt") as fa, gzip.open(os.path.join(DATA, f"hp_{CHK}.txt.gz"), "rt") as fb:
    for la, lb in zip(fa, fb):
        pa, pb = la.split(), lb.split()
        i = int(pa[0]); assert i == int(pb[0])
        if i > N:
            break
        dd = max(abs(mpmath.mpf(u) - mpmath.mpf(v)) for u, v in zip(pa[1:], pb[1:]))
        if dd > maxd:
            maxd, at = dd, i
        if i == window_steps:
            maxd_window = maxd
print(f"ref check: max |{CHK}-{REF}| over chart window (0..{window_steps*DT:.3f} s) = {mpmath.nstr(maxd_window, 3)}; "
      f"over full {N*DT:.0f} s = {mpmath.nstr(maxd, 3)} at t={at*DT:.3f}")
res["reference_check"] = {"window_s": window_steps * DT, "max_abs_diff_window": float(maxd_window),
                          "max_abs_diff_full": float(maxd), "passes_1e-20": bool(maxd_window < mpmath.mpf("1e-20"))}
json.dump(res, open(os.path.join(DATA, "measurements.json"), "w"), indent=2)
