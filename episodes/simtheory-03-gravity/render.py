"""SimTheory #03 - frame renderer, built on the series style kit (simtheory-style/simtheory_style.py).

Every colour, glow, size, font, text style and export setting comes from the kit (role colours, GLOW bloom,
SIZES, TEXT styles, EXPORT). This file only decides WHAT is drawn WHERE and WHEN. Every pendulum angle is
evaluated from the RK4 trajectories in sim.py.

  python3 render.py --still 0 14.15 ...     -> build/stills/*.png
  python3 render.py --video                 -> build/video.mp4 (silent, kit export settings)
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "simtheory-style"))
import simtheory_style as S  # noqa: E402

import sim  # noqa: E402
import timeline as TL  # noqa: E402

BUILD = os.path.join(HERE, "build")
W, H = 1440, 2560                # brief: 1440x2560; the kit scales every size by H/1920
SC = H / 1920.0

# --- safe area from the brief (clear of top 15 %, bottom 35 %, right 20 %) --------------------
SAFE_X1, SAFE_Y0, SAFE_Y1 = int(W * 0.80), int(H * 0.15), int(H * 0.65)    # 1152, 384, 1664
CX = 576                         # content centre line (centre of 0..1152)
TEXT_MAXW = 1056                 # 48..1104

# --- kit colour roles ----------------------------------------------------------------------
EARTH = S.role_color("main")            # cyan: the normal / ideal case (Earth)
HEAVY = S.role_color("perturbed")       # orange: the "different" case (10x gravity)
GHOST = S.role_color("extra", 0)        # purple: extra case (100x ghost)
REF = S.role_color("reference")         # white: the true result (formula)
HUD = S.C("hud")
EARTH_N, HEAVY_N, GHOST_N = "cyan", "orange", "purple"   # role colour names for kit text styles
# counters use the kit "number" style, glowing in their own role colour
S.TEXT["number_main"] = dict(S.TEXT["number"], color=EARTH_N, glow=EARTH_N)
S.TEXT["number_perturbed"] = dict(S.TEXT["number"], color=HEAVY_N, glow=HEAVY_N)

# --- layout (px at 1440x2560) ----------------------------------------------------------------
CAP_Y = 468                      # centre of the first caption line
CAP_SCALE = 0.72                 # kit "headline" at 0.72 -> ~100 px, two lines max
LABEL_Y = 760
LABEL_SCALE = 1.2
COUNTER_Y = 872
COUNTER_SCALE = 2.45
PIVOT_Y = 992
PIV_E = (312.0, PIVOT_Y)
PIV_H = (840.0, PIVOT_Y)
PIV_G = (float(CX), PIVOT_Y)
LPX = 520.0                      # 1 m on screen
FORMULA_C = (float(CX), 1262.0)
FORMULA_W = 930.0
FIELD_Y = (720.0, 1655.0)
NEBULA_C = (CX, 1180)

# caption line breaks (text itself is unchanged from the brief)
BREAKS = {
    "10× stronger gravity…": ["10× stronger", "gravity…"],
    "…10× faster swing?": ["…10× faster", "swing?"],
    "Gravity hides under a square root.": ["Gravity hides", "under a square root."],
    "For 10× faster, you'd need 100× gravity.": ["For 10× faster, you'd", "need 100× gravity."],
}

# ---------------------------------------------------------------------------------------------
E = sim.Traj(sim.G_EARTH, 13.0)
HV = sim.Traj(sim.G_HEAVY, 13.0)
GH = sim.Traj(sim.G_GHOST, 7.0)
TH0 = sim.THETA0
EARTH_SYNC_TH = float(E.theta(TL.SYNC_SIM))
T_GHOST = sim.exact_period(sim.G_GHOST)


def theta_e(t):
    """Displayed Earth angle (rad) at video time t (array ok)."""
    t = np.asarray(t, float)
    th = E.theta(np.clip(t - TL.T_REL, 0.0, TL.SYNC_SIM))
    # held after the freeze: the 0.03 deg gap to the release angle (Earth reached it 16.8 ms earlier)
    # is closed invisibly while frozen, so the loop lands exactly on frame 1
    corr = np.clip((t - (TL.T_SYNC + 2.0)) / 4.0, 0, 1)
    corr = corr * corr * (3 - 2 * corr)
    return np.where(t > TL.T_SYNC, th + (TH0 - EARTH_SYNC_TH) * corr, th)


def theta_h(t):
    return HV.theta(np.clip(np.asarray(t, float) - TL.T_REL, 0.0, TL.SYNC_SIM))


def theta_g(t):
    return GH.theta(np.clip(np.asarray(t, float) - TL.GHOST_REL, 0.0, 6.9))


def bob_xy(piv, th):
    return piv[0] + LPX * np.sin(th), piv[1] + LPX * np.cos(th)


# =============================================================================================
class Cam:
    """Slow push-in about the formula. World -> screen; sizes scale with the zoom."""

    def __init__(self, z, c=FORMULA_C):
        self.z, self.c = z, c

    def p(self, x, y):
        return self.c[0] + (np.asarray(x) - self.c[0]) * self.z, self.c[1] + (np.asarray(y) - self.c[1]) * self.z

    def pts(self, xs, ys):
        X, Y = self.p(xs, ys)
        return np.stack([np.atleast_1d(X), np.atleast_1d(Y)], 1)


def ring_pts(c, R, n=96):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a)], 1)


# ---- formula (real KaTeX masks, drawn like kit text: crisp overlay + glow layer) ---------------
_FM = None


def formula_masks():
    global _FM
    if _FM is None:
        d = os.path.join(BUILD, "formula")
        out = {n: np.asarray(Image.open(os.path.join(d, n + ".png")), np.float32)[..., 3] / 255.0
               for n in ["all", "T", "eq", "twopi", "root", "L", "slash", "g"]}
        ys, xs = np.nonzero(out["all"] > 0.03)
        pad = 130
        y0, y1, x0, x1 = ys.min() - pad, ys.max() + 1 + pad, xs.min() - pad, xs.max() + 1 + pad
        full = np.zeros((out["all"].shape[0] + 2 * pad, out["all"].shape[1] + 2 * pad), np.float32)
        for n in list(out):
            f2 = full.copy()
            f2[pad:pad + out[n].shape[0], pad:pad + out[n].shape[1]] = out[n]
            out[n] = np.ascontiguousarray(f2[y0 + pad:y1 + pad, x0 + pad:x1 + pad])
        out["_inkw"] = float(xs.max() - xs.min() + 1)
        _FM = out
    return _FM


def place(mask, cx, cy, scale):
    """Resample mask centred at (cx, cy) with uniform scale (sub-pixel exact). Returns cov, X, Y."""
    mh, mw = mask.shape
    hw, hh = mw * scale / 2, mh * scale / 2
    x0, x1 = int(math.floor(cx - hw)) - 1, int(math.ceil(cx + hw)) + 1
    y0, y1 = int(math.floor(cy - hh)) - 1, int(math.ceil(cy + hh)) + 1
    a = 1.0 / scale
    c0 = (x0 - (cx - hw)) / scale + 0.5 * a - 0.5
    f0 = (y0 - (cy - hh)) / scale + 0.5 * a - 0.5
    out = Image.fromarray(mask, "F").transform((x1 - x0, y1 - y0), Image.AFFINE, (a, 0, c0 + 0.5, 0, a, f0 + 0.5),
                                               resample=Image.BICUBIC)
    return np.clip(np.asarray(out, np.float32), 0, 1), x0, y0


def draw_formula_part(f, cov, X, Y, color, alpha, glow_color, glow_gain):
    rgb = np.broadcast_to(color, cov.shape + (3,))
    f._paste_overlay(np.dstack([rgb, cov * alpha]), X, Y)
    if glow_gain > 0:
        g = S.gaussian_blur(cov, 22 * f.s)
        f._add_glow(g[..., None] * glow_color * (0.9 * glow_gain * alpha), X, Y)


# ---- gravity field + sparks -------------------------------------------------------------------
_rng = np.random.default_rng(3)


def _field(n, x_lo, x_hi, wraps):
    return (_rng.uniform(x_lo, x_hi, n), _rng.uniform(0, 1, n), _rng.choice(wraps, n), _rng.uniform(0.45, 1.0, n))


# speeds are whole numbers of wraps per loop, so the field loops seamlessly
FIELD_E = _field(30, 70, 545, [3, 4, 4, 5])
FIELD_H = _field(78, 607, 1100, [14, 16, 17, 18, 20])


def draw_field(f, cam, t, field, color, alpha, streak_s):
    xs, y0, k, b = field
    hgt = FIELD_Y[1] - FIELD_Y[0]
    v = k * hgt / TL.DUR
    ph = (y0 + k * (t / TL.DUR)) % 1.0
    ys = FIELD_Y[0] + ph * hgt
    edge = np.clip(np.minimum(ph, 1 - ph) / 0.12, 0, 1)
    a = b * edge * edge * (3 - 2 * edge)
    lv = np.round(a * 4) / 4                      # 4 brightness buckets -> one kit stamp per bucket
    for q in (0.25, 0.5, 0.75, 1.0):
        sel = np.nonzero(lv == q)[0]
        if len(sel) == 0:
            continue
        polys = [cam.pts([xs[i], xs[i]], [ys[i] - max(v[i] * streak_s, 1.5), ys[i]]) for i in sel]
        f._stamp(polys, S.SIZES["spark"] * 0.9 * cam.z, color, alpha * q)


SPARK_RATE = 34.0
SPARK_LIFE = 0.85


def _sparks(seed):
    n = int(TL.DUR * SPARK_RATE)
    r = np.random.default_rng(seed)
    ang = r.uniform(0, 2 * math.pi, n)
    sp = r.uniform(15, 75, n)
    return (np.arange(n) / SPARK_RATE, np.cos(ang) * sp, np.sin(ang) * sp - 20, r.uniform(0.35, 1.0, n),
            r.uniform(-14, 14, (n, 2)))


SPARKS_E = _sparks(11)
SPARKS_H = _sparks(12)


def draw_sparks(f, cam, t, sparks, bobfn, color, acc, alpha):
    """glow sparks shed by the bob; they fall with (a cosmetic multiple of) each side's gravity"""
    ts, vx, vy, br, off = sparks
    buckets = {}
    for wrap in (0.0, -TL.DUR):                  # periodic in DUR -> seamless loop
        age = t - (ts + wrap)
        sel = np.nonzero((age >= 0) & (age < SPARK_LIFE))[0]
        if len(sel) == 0:
            continue
        bx, by = bobfn(ts[sel])
        for j, i in enumerate(sel):
            a = age[i]
            x = bx[j] + off[i, 0] + vx[i] * a
            y = by[j] + off[i, 1] + vy[i] * a + 0.5 * acc * a * a
            q = round(br[i] * (1 - a / SPARK_LIFE) ** 2 * 4) / 4
            if q > 0:
                X, Y = cam.p(x, y)
                buckets.setdefault(q, []).append((float(X), float(Y)))
    for q, xy in buckets.items():
        f.dots(xy, color, r=S.SIZES["spark"] * cam.z, alpha=alpha * q * 0.6)


# ---- pendulum ---------------------------------------------------------------------------------
def draw_pendulum(f, cam, t, piv, thfn, color, dim, trail_s, trail_fade, shutter=0.5, max_sub=8):
    span = shutter / TL.FPS
    th_lo, th_hi = thfn(np.array([t - span / 2, t + span / 2]))
    n = int(min(max_sub, max(1, math.ceil(abs(th_hi - th_lo) * LPX / 3.0))))
    ts = t + (np.arange(n) / max(n - 1, 1) - 0.5) * span if n > 1 else np.array([t])
    P = cam.pts([piv[0]], [piv[1]])[0]
    # mount bar
    f.line(cam.pts([piv[0] - 40, piv[0] + 40], [piv[1], piv[1]]), color, width=S.SIZES["line"] * 1.3 * cam.z,
           alpha=0.55 * dim)
    # kit trail (oldest -> newest)
    if trail_s > 0 and trail_fade > 0:
        tt = t - np.linspace(trail_s, 0, 64)
        bx, by = bob_xy(piv, thfn(tt))
        if np.hypot(bx[0] - bx[-1], by[0] - by[-1]) > 1.0:
            f.trail(cam.pts(bx, by), color, alpha=trail_fade * dim)
    for th in thfn(ts):
        bx, by = bob_xy(piv, th)
        B = cam.pts([bx], [by])[0]
        f.line([P, B], color, width=S.SIZES["line"] * cam.z, alpha=dim / n)
        f.bob(B[0], B[1], color, r=S.SIZES["bob"] * cam.z, alpha=dim / n)
    f.pivot(P[0], P[1], alpha=dim)


def pop_scale(t, events):
    s = 0.0
    for te in events:
        u = t - te
        if 0 <= u < 0.4:
            s += 0.30 * (u / 0.045) * math.exp(1 - u / 0.045)
    return 1.0 + s


def caption_state(t):
    out = []
    for text, a, b in TL.CAPTIONS:
        if t < a or t >= b + TL.CAP_FADE_OUT:
            continue
        if a > 0 and t < a + TL.CAP_FADE_IN:
            u = TL.smoother((t - a) / TL.CAP_FADE_IN)
            out.append((text, u, (1 - u) * 16))
        elif t >= b:
            u = TL.smoother((t - b) / TL.CAP_FADE_OUT)
            out.append((text, 1 - u, -u * 10))
        else:
            out.append((text, 1.0, 0.0))
    return out


# =============================================================================================
def render(fi):
    t = fi / TL.FPS
    cam = Cam(float(TL.cam_zoom(t)))
    f = S.Frame(w=W, h=H, index=fi, nebula_center=NEBULA_C)
    frozen = t >= TL.T_SYNC
    since_sync = t - TL.T_SYNC

    dim = 1.0 - 0.80 * TL.ramp(t, TL.T_FORMULA - 0.3, TL.T_FORMULA + 0.4) + 0.80 * TL.ramp(t, 26.6, 27.6)
    if TL.GHOST_IN[0] - 0.2 <= t < 26.6:
        dim = min(dim, 0.20)
    field_a = 1.0 - 0.5 * TL.ramp(t, TL.T_FORMULA - 0.3, TL.T_FORMULA + 0.4) + 0.5 * TL.ramp(t, 26.6, 27.6)

    # gravity field (behind): falls faster and denser on the 10x side
    draw_field(f, cam, t, FIELD_E, EARTH, 0.30 * field_a, 0.035)
    draw_field(f, cam, t, FIELD_H, HEAVY, 0.30 * field_a, 0.035)

    # divider (kit HUD colour)
    top, bot = 715.0, 1655.0
    for k in range(20):
        ya, yb = top + (bot - top) * k / 20, top + (bot - top) * (k + 1) / 20
        a = math.sin(math.pi * (k + 0.5) / 20) ** 0.8
        f.line(cam.pts([CX, CX], [ya, yb]), HUD, width=S.SIZES["ensemble_line"] * 1.6 * cam.z, alpha=0.35 * a)

    draw_sparks(f, cam, t, SPARKS_E, lambda ts: bob_xy(PIV_E, theta_e(ts)), EARTH, 45.0, dim)
    draw_sparks(f, cam, t, SPARKS_H, lambda ts: bob_xy(PIV_H, theta_h(ts)), HEAVY, 450.0, dim)

    trail_fade = 1.0 if not frozen else max(0.0, 1 - since_sync / 0.7)
    draw_pendulum(f, cam, t, PIV_E, theta_e, EARTH, dim, 0.55, trail_fade)
    draw_pendulum(f, cam, t, PIV_H, theta_h, HEAVY, dim, 0.30, trail_fade)

    # ring at each full swing; a bigger one at the sync
    for evs, piv, thfn, col in ((TL.CNT_E, PIV_E, theta_e, EARTH), (TL.CNT_H, PIV_H, theta_h, HEAVY)):
        for i, te in enumerate(evs):
            u = t - te
            last = i == len(evs) - 1
            dur = 0.9 if last else 0.38
            if 0 <= u < dur:
                v = u / dur
                c = bob_xy(piv, thfn(te))
                R = S.SIZES["bob"] * SC * 1.6 + (90 if last else 40) * (1 - (1 - v) ** 2.2)
                pts = ring_pts(c, R)
                f.line(cam.pts(pts[:, 0], pts[:, 1]), col, width=S.SIZES["line"] * (1.3 if last else 1.0) * cam.z,
                       alpha=(1 - v) ** 1.5 * (1.0 if last else 0.6))

    # ghost (100x gravity): real sim, 360-degree shutter + one-period long exposure -> blur
    ga = TL.ghost_alpha(t)
    if ga > 0:
        shim = 0.92 + 0.08 * math.sin(fi * 2.1) * math.sin(fi * 0.77)
        a = TL.GHOST_ALPHA * ga * shim
        P = cam.pts([PIV_G[0]], [PIV_G[1]])[0]
        f.line(cam.pts([PIV_G[0] - 40, PIV_G[0] + 40], [PIV_G[1]] * 2), GHOST, width=S.SIZES["line"] * 1.3 * cam.z,
               alpha=0.5 * a)
        n = 24
        for th in theta_g(t + (np.arange(n) / (n - 1) - 0.5) / TL.FPS):
            B = cam.pts(*[[v] for v in bob_xy(PIV_G, th)])[0]
            f.line([P, B], GHOST, width=S.SIZES["line"] * 1.2 * cam.z, alpha=2.2 * a / n)
            f.bob(B[0], B[1], GHOST, r=S.SIZES["bob"] * cam.z, alpha=2.2 * a / n)
        nl = 32
        for th in theta_g(t - np.arange(1, nl + 1) / nl * T_GHOST):
            B = cam.pts(*[[v] for v in bob_xy(PIV_G, th)])[0]
            f.line([P, B], GHOST, width=S.SIZES["line"] * 1.5 * cam.z, alpha=1.1 * a / nl)
            f.bob(B[0], B[1], GHOST, r=S.SIZES["bob"] * cam.z, alpha=1.4 * a / nl)
        f.pivot(P[0], P[1], alpha=a)
        X, Y = cam.p(CX, 1612)
        f.text("100× GRAVITY", float(X), float(Y), style="label", color=GHOST_N, alpha=min(1.0, 1.4 * a),
               scale=float(LABEL_SCALE * cam.z))

    # labels + counters (world space, kit "label" and "number" styles in role colours)
    lab_a = 1.0 - 0.35 * TL.ramp(t, TL.GHOST_IN[0] - 0.2, TL.GHOST_IN[0] + 0.3) + 0.35 * TL.ramp(t, 26.6, 27.4)
    for text, x, col in (("EARTH · 1g", PIV_E[0], EARTH_N), ("10× GRAVITY", PIV_H[0], HEAVY_N)):
        X, Y = cam.p(x, LABEL_Y)
        f.text(text, float(X), float(Y), style="label", color=col, alpha=lab_a, scale=float(LABEL_SCALE * cam.z))
    for evs, x, style in ((TL.CNT_E, PIV_E[0], "number_main"), (TL.CNT_H, PIV_H[0], "number_perturbed")):
        if t < TL.T_RESET:
            n = int(np.sum(evs <= t + 1e-9))
            ps = pop_scale(t, evs)
        else:
            n = 0
            ps = pop_scale(t, [TL.T_RESET])
        X, Y = cam.p(x, COUNTER_Y)
        f.text(str(n), float(X), float(Y), style=style, alpha=lab_a, scale=float(COUNTER_SCALE * ps * cam.z))

    # formula: KaTeX, reference white with the kit's cyan headline glow; g in the 10x-gravity orange
    if TL.T_FORMULA <= t < TL.T_FORMULA_OUT[1]:
        fm = formula_masks()
        sc = FORMULA_W / fm["_inkw"] * cam.z
        out_a = 1 - TL.ramp(t, *TL.T_FORMULA_OUT)
        for name, dt in TL.FORMULA_GROUPS:
            u = (t - TL.T_FORMULA - dt) / 0.38
            if u <= 0:
                continue
            e = TL.smoother(u)
            a = e * out_a
            cx, cy = cam.p(FORMULA_C[0], FORMULA_C[1] + (1 - e) * 26)
            cov, X, Y = place(fm[name], float(cx), float(cy), sc * (0.94 + 0.06 * e))
            col, glow_c, gain = REF, EARTH, 0.55
            if name == "g":
                col, glow_c = HEAVY, HEAVY
            if name == "root" and t >= TL.T_ROOT_GLOW:
                ph = t - TL.T_ROOT_GLOW
                pulse = 0.5 - 0.5 * math.cos(2 * math.pi * ph / 0.9)
                rise = TL.smooth(ph / 0.5)
                gain = 0.55 + rise * (1.6 + 2.8 * pulse)
                # the radical itself also becomes light, so the kit bloom makes it pulse
                h, w = cov.shape
                x0, y0 = max(X, 0), max(Y, 0)
                x1, y1 = min(X + w, W), min(Y + h, H)
                f.light[y0:y1, x0:x1] += cov[y0 - Y:y1 - Y, x0 - X:x1 - X, None] * EARTH * (
                    rise * (0.9 + 1.8 * pulse) * out_a)
            draw_formula_part(f, cov, X, Y, col, a, glow_c, gain)

    # captions (screen space): kit "headline" style
    for text, a, dy in caption_state(t):
        lines = BREAKS.get(text, [text])
        lh = S.TEXT["headline"]["size"] * CAP_SCALE * SC * 1.08
        for i, ln in enumerate(lines):
            f.text(ln, CX, CAP_Y + dy + i * lh, style="headline", alpha=a, scale=CAP_SCALE, max_width=TEXT_MAXW)
    return f.finish()


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs="*", type=float)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(BUILD, "video.mp4"))
    args = ap.parse_args()
    if args.still is not None:
        os.makedirs(os.path.join(BUILD, "stills"), exist_ok=True)
        for ts in args.still:
            fi = int(round(ts * TL.FPS))
            p = os.path.join(BUILD, "stills", f"f{fi:04d}.png")
            Image.fromarray(render(fi)).save(p)
            print(p)
    if args.video:
        S.render_video(render, TL.NFRAMES, args.out, workers=os.cpu_count(), chunksize=2)
        print("colour tags:", S.check_export(args.out))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
