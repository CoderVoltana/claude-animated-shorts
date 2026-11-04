"""Step 1: run the full simulation + all measurements, save every state to disk (data/sim.npz, data/measure.json,
data/frames.csv). Rendering and audio read only these files."""
import json, os, csv
import numpy as np
from physics import rk4_step, energy, E_REST, wrap180
import timeline as TL

DT = 1.0 / 1200.0              # 0.833 ms (divides the 1/60 s frame exactly)
THETA0 = 120.0
OFFSET_DEG = 0.002             # twin offset on theta1 (same in both lanes)
C_VISC = 0.03                  # viscous joint friction  [N m s / rad]
MU_DRY = 0.20                  # dry (Coulomb) joint friction [N m], regularised: mu*tanh(w/W0)
W0 = 0.05                      # regularisation speed [rad/s]
T_SIM = 112.0                  # sim seconds (covers the whole forward timeline with margin)
OUT = os.path.join(TL.HERE, "data")
os.makedirs(OUT, exist_ok=True)

# pendulum order: 0 = left twin A, 1 = left twin B (offset), 2 = right twin A, 3 = right twin B (offset)
c = np.array([0.0, 0.0, C_VISC, C_VISC])
mu = np.array([0.0, 0.0, MU_DRY, MU_DRY])
s = np.zeros((4, 4))
s[:, :2] = np.radians(THETA0)
s[[1, 3], 0] += np.radians(OFFSET_DEG)
n = int(round(T_SIM / DT))
S = np.empty((n + 1, 4, 4))
S[0] = s
for i in range(n):
    s = rk4_step(s, c, DT, mu, W0)
    S[i + 1] = s
t = np.arange(n + 1) * DT
E = energy(S)                                        # (n+1, 4)
E0 = E[0]
scale = E0 - E_REST                                  # available energy above rest (44.145 J)
efrac = (E - E_REST) / scale


def sep_deg(a, b):
    d1 = np.abs(wrap180(np.degrees(S[:, a, 0] - S[:, b, 0])))
    d2 = np.abs(wrap180(np.degrees(S[:, a, 1] - S[:, b, 1])))
    return d1, d2


def flips(p):
    """flip = an arm passes over the top (its angle crosses an odd multiple of 180 deg)"""
    ev = []
    for arm in (0, 1):
        k = np.floor((S[:, p, arm] + np.pi) / (2 * np.pi))
        idx = np.nonzero(np.diff(k) != 0)[0] + 1
        ev += [(float(t[i]), arm + 1, int(np.sign(k[i] - k[i - 1]))) for i in idx]
    return sorted(ev)


sepL1, sepL2 = sep_deg(0, 1)
sepR1, sepR2 = sep_deg(2, 3)
sepR = np.maximum(sepR1, sepR2)
sepL = np.maximum(sepL1, sepL2)
speedR = np.degrees(np.abs(S[:, 2:4, 2:4])).max(axis=(1, 2))
fl = {p: flips(p) for p in range(4)}

# ---- key measured times (sim seconds)
def first_above(x, thr):
    i = np.nonzero(x > thr)[0]
    return float(t[i[0]]) if len(i) else None


def last_above(x, thr):
    i = np.nonzero(x >= thr)[0]
    return float(t[i[-1] + 1]) if len(i) and i[-1] + 1 < len(t) else None


m = {}
m["split20_right_sim"] = first_above(sepR2, 20.0)
m["split20_left_sim"] = first_above(sepL2, 20.0)
m["split_vis_right_sim"] = first_above(sepR2, 5.0)
m["split_vis_left_sim"] = first_above(sepL2, 5.0)
m["last_flip_right_sim"] = max(e[0] for p in (2, 3) for e in fl[p])
m["remerge_right_sim"] = last_above(sepR, 0.5)
m["rest_right_sim"] = last_above(speedR, 0.5)

# ---- solve RMAX so the re-merge lands at video MERGE_TARGET (bisection on the closed-form map)
lo, hi = 1.0, 30.0
for _ in range(100):
    mid = 0.5 * (lo + hi)
    if TL.sim_forward(TL.MERGE_TARGET, mid) < m["remerge_right_sim"]:
        lo = mid
    else:
        hi = mid
rmax = round(0.5 * (lo + hi), 4)
json.dump({"rmax": rmax}, open(os.path.join(OUT, "timemap.json"), "w"))
s_end = TL.sim_end(rmax)
assert s_end < T_SIM - 1.0, (s_end, T_SIM)


def v_of_s(ss):
    """invert the forward map (monotone) -> video time"""
    if ss is None:
        return None
    a, b = TL.T_REL, TL.T_RW0
    for _ in range(80):
        mid = 0.5 * (a + b)
        if TL.sim_forward(mid, rmax) < ss:
            a = mid
        else:
            b = mid
    return round(0.5 * (a + b), 4)


for k in list(m):
    m[k.replace("_sim", "_video")] = v_of_s(m[k])

# ---- checks over the part of the run that is shown (0 .. s_end)
iend = int(np.ceil(s_end / DT)) + 1
drift = np.abs(E[:iend, :2] - E0[:2]) / np.abs(E0[:2])
drift_avail = np.abs(E[:iend, :2] - E0[:2]) / scale[:2]
dE = np.diff(E[:iend, 2:], axis=0) / scale[2:]
fin = S[iend - 1]
m.update({
    "dt_s": DT, "integrator": "RK4 float64", "theta0_deg": THETA0, "twin_offset_deg": OFFSET_DEG,
    "friction_model": "per joint torque = -c*w_rel - mu*tanh(w_rel/w0); joint1 w_rel = w1, joint2 w_rel = w2 - w1",
    "c_viscous_Nms": C_VISC, "mu_dry_Nm": MU_DRY, "w0_rad_s": W0,
    "rmax_timelapse": rmax, "sim_end_s": s_end,
    "E0_J": float(E0[0]), "E_rest_J": E_REST, "E_available_J": float(scale[0]),
    "max_energy_drift_left_rel_E0": float(drift.max()),
    "max_energy_drift_left_rel_available": float(drift_avail.max()),
    "max_energy_rise_right_rel_per_step": float(dE.max()),
    "final_sep_left_deg": float(sepL[iend - 1]), "final_sep_left_bottom_deg": float(sepL2[iend - 1]),
    "final_sep_right_deg": float(sepR[iend - 1]),
    "final_right_angles_deg": [float(x) for x in wrap180(np.degrees(fin[2:4, :2])).ravel()],
    "final_right_max_speed_deg_s": float(speedR[iend - 1]),
    "final_energy_left_pct": [float(100 * efrac[iend - 1, 0]), float(100 * efrac[iend - 1, 1])],
    "final_energy_right_pct": [float(100 * efrac[iend - 1, 2]), float(100 * efrac[iend - 1, 3])],
    "flips_left": [len(fl[0]), len(fl[1])], "flips_right": [len(fl[2]), len(fl[3])],
    "flip_events_right": {"A": fl[2], "B": fl[3]},
})
checks = {
    "no_friction_energy_drift_lt_0.1pct": m["max_energy_drift_left_rel_E0"] < 1e-3,
    "friction_energy_never_increases_lt_1e-9_per_step": m["max_energy_rise_right_rel_per_step"] < 1e-9,
    "friction_twins_final_sep_lt_0.1deg": m["final_sep_right_deg"] < 0.1,
    "friction_twins_hanging_down_lt_0.1deg": max(abs(x) for x in m["final_right_angles_deg"]) < 0.1,
    "split_right_gt20_by_8s": m["split20_right_video"] is not None and m["split20_right_video"] <= 8.0,
    "split_left_gt20_by_9s": m["split20_left_video"] is not None and m["split20_left_video"] <= 9.0,
    "flips_stop_10_12s": 10.0 <= m["last_flip_right_video"] <= 12.0,
    "remerge_20_22s": 20.0 <= m["remerge_right_video"] <= 22.0,
    "rest_by_24s": m["rest_right_video"] <= 24.0,
}
m["checks"] = checks
json.dump(m, open(os.path.join(OUT, "measure.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "sim.npz"), S=S[:iend + 200], t=t[:iend + 200], E=E[:iend + 200], dt=DT)

# ---- per-frame measurement log
ft = TL.frame_times()
fs = TL.sim_time(ft, rmax)
idx = np.clip(np.round(fs / DT).astype(int), 0, len(t) - 1)
with open(os.path.join(OUT, "frames.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "video_t", "sim_t", "rate", "sepL_bottom_deg", "sepR_bottom_deg", "sepL_max_deg", "sepR_max_deg",
                "E_LA_J", "E_LB_J", "E_RA_J", "E_RB_J", "energy_left_pct", "energy_right_pct", "flips_L_total", "flips_R_total"])
    allL = sorted(e[0] for p in (0, 1) for e in fl[p]); allR = sorted(e[0] for p in (2, 3) for e in fl[p])
    for k in range(TL.NFR):
        i = idx[k]
        w.writerow([k, f"{ft[k]:.4f}", f"{fs[k]:.5f}", f"{float(TL.rate(ft[k], rmax)) if ft[k] < TL.T_RW0 else -1:.4f}",
                    f"{sepL2[i]:.5f}", f"{sepR2[i]:.5f}", f"{sepL[i]:.5f}", f"{sepR[i]:.5f}",
                    *[f"{E[i, p]:.8f}" for p in range(4)],
                    f"{100 * efrac[i, :2].mean():.4f}", f"{100 * efrac[i, 2:].mean():.4f}",
                    int(np.searchsorted(allL, fs[k], side='right')), int(np.searchsorted(allR, fs[k], side='right'))])
print(json.dumps({k: v for k, v in m.items() if k != "flip_events_right"}, indent=1))
