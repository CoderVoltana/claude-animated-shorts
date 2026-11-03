"""SimTheory #02 renderer: Skia (CPU) -> raw RGB frames -> ffmpeg (lossless FFV1 segments).

    python render.py --still 16.6 out.png          one frame
    python render.py --seg 0 450 seg0.mkv           frames [0, 450) to a lossless segment

All sizes are authored for 1440x2560 (final). Every pendulum position comes from sim.npz (no animation fakery):
GHOST = forward record played backwards, CYAN / ORANGE = integrated reversed copies.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import skia

import timeline as TL
from config import W, H, FPS, T_SIM
from physics import DT, L, all_omegas

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/usr/share/fonts/opentype/inter/"

# ------------------------------------------------------------------ series style (identical every episode)
CYAN = (0, 190, 255)          # cyan + orange sum to ~neutral white while aligned (additive)
CYAN_CORE = (140, 228, 255)
ORANGE = (255, 98, 0)
ORANGE_CORE = (255, 178, 112)
PURPLE = (190, 70, 255)
PURPLE_CORE = (230, 185, 255)
WHITE = (255, 255, 255)
TAG = "SIM #02"

# ------------------------------------------------------------------ layout (safe zone: x < 1224, y < 2048)
SAFE_R = int(W * 0.85)            # 1224
SAFE_B = int(H * 0.80)            # 2048
CX = SAFE_R / 2                    # 612: centre of the usable width
PIV = (CX, 1420.0)                 # pendulum pivot on screen
RPX = 270.0                        # px per metre at zoom 1 (2 m reach = 540 px)
CAP_Y = 640.0                      # caption block centre
GRAPH = (84.0, 790.0, 1140.0, 1330.0)

# ------------------------------------------------------------------ data
D = np.load(os.path.join(HERE, "sim.npz"))
F, C, O = D["F"], D["C"], D["O"]
N = len(F) - 1
ANG_C, ANG_O = D["ang_c"], D["ang_o"]
INFO = TL.INFO
FIT = INFO["lyapunov_fit"]


def _bobs(Y):
    x1 = L * np.sin(Y[:, 0]); y1 = -L * np.cos(Y[:, 0])
    return np.stack([x1, y1, x1 + L * np.sin(Y[:, 1]), y1 - L * np.cos(Y[:, 1])], 1)


PF, PC, PO = _bobs(F), _bobs(C), _bobs(O)
ARC = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(PF[:, 2]), np.diff(PF[:, 3])))])  # ghost path arc length [m]


def font(name, size):
    return skia.Font(skia.Typeface.MakeFromFile(FONT_DIR + name), size)


F_CAP = font("InterDisplay-Bold.otf", 92)
F_TAG = font("InterDisplay-Light.otf", 50)
F_HUD = font("Inter-Medium.otf", 50)
F_HUD_S = font("Inter-SemiBold.otf", 40)
F_HUD_L = font("Inter-Medium.otf", 30)
F_G = font("Inter-SemiBold.otf", 38)
F_G_S = font("Inter-Medium.otf", 33)
for f_ in (F_CAP, F_TAG, F_HUD, F_HUD_S, F_HUD_L, F_G, F_G_S):
    f_.setSubpixel(True)
    f_.setEdging(skia.Font.Edging.kAntiAlias)


# ------------------------------------------------------------------ helpers
def sstep(a, b, t):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * u * (u * (6 * u - 15) + 10)


def col(c, a=1.0):
    return skia.Color4f(c[0] / 255, c[1] / 255, c[2] / 255, max(0.0, min(1.0, a)))


def paint(c, a=1.0, stroke=None, blend=skia.BlendMode.kPlus, cap=skia.Paint.kRound_Cap, blur=0.0):
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(col(c, a))
    p.setBlendMode(blend)
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap(cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def zoom(t):
    """slow push-in during the rewind, easing back out at the reveal."""
    return 1.0 + 0.07 * sstep(11.0, 22.0, t) - 0.07 * sstep(23.6, 25.8, t)


class Cam:
    def __init__(self, t):
        self.z = zoom(t)
        self.s = RPX * self.z

    def xy(self, x, y):
        return PIV[0] + self.s * x, PIV[1] - self.s * y


# ------------------------------------------------------------------ pendulum drawing (all additive)
def draw_copy(cv, cam, P, kind, a):
    if a <= 0.002:
        return
    x0, y0 = PIV
    x1, y1 = cam.xy(P[0], P[1])
    x2, y2 = cam.xy(P[2], P[3])
    path = skia.Path()
    path.moveTo(x0, y0); path.lineTo(x1, y1); path.lineTo(x2, y2)
    z = cam.z
    if kind == "ghost":
        cv.drawPath(path, paint(WHITE, 0.09 * a, stroke=32 * z))
        cv.drawPath(path, paint(WHITE, 0.36 * a, stroke=11 * z))
        for (bx, by) in ((x1, y1), (x2, y2)):
            cv.drawCircle(bx, by, 25 * z, paint(WHITE, 0.18 * a))
            cv.drawCircle(bx, by, 25 * z, paint(WHITE, 0.55 * a, stroke=5 * z))
        return
    c, core = (CYAN, CYAN_CORE) if kind == "cyan" else (ORANGE, ORANGE_CORE)
    cv.drawPath(path, paint(c, 0.22 * a, stroke=30 * z))
    cv.drawPath(path, paint(c, 0.75 * a, stroke=13 * z))
    cv.drawPath(path, paint(core, 0.55 * a, stroke=5 * z))
    for (bx, by) in ((x1, y1), (x2, y2)):
        cv.drawCircle(bx, by, 27 * z, paint(c, 0.85 * a))
        cv.drawCircle(bx, by, 14 * z, paint(core, 0.65 * a))


def draw_pivot(cv, cam):
    x, y = PIV
    cv.drawLine(x - 46, y, x + 46, y, paint((140, 140, 150), 0.55, stroke=6))
    cv.drawCircle(x, y, 11, paint((150, 150, 160), 0.8))


def draw_trail(cv, cam, pts, c, width, a, power=1.6):
    """pts: (n, 2) world positions oldest -> newest. Drawn inside a 'lighten' layer so overlaps don't bead."""
    n = len(pts)
    if n < 2 or a <= 0.002:
        return
    lp = skia.Paint()
    lp.setBlendMode(skia.BlendMode.kPlus)
    cv.saveLayer(None, lp)                                     # layer composited back additively
    S = np.empty_like(pts)
    S[:, 0] = PIV[0] + cam.s * pts[:, 0]
    S[:, 1] = PIV[1] - cam.s * pts[:, 1]
    chunks = 28
    edges = np.linspace(0, n - 1, chunks + 1).astype(int)
    for i in range(chunks):
        i0, i1 = edges[i], edges[i + 1]
        if i1 <= i0:
            continue
        u = (i + 1) / chunks
        al = a * u ** power
        w = width * cam.z * (0.35 + 0.65 * u)
        p = skia.Path()
        p.moveTo(*S[i0])
        for k in range(i0 + 1, i1 + 1):
            p.lineTo(*S[k])
        cv.drawPath(p, paint(c, al * 0.30, stroke=w * 3.0, blend=skia.BlendMode.kLighten))
        cv.drawPath(p, paint(c, al, stroke=w, blend=skia.BlendMode.kLighten))
    cv.restore()


def draw_ghost_path(cv, cam, k_now, a):
    """faint dashed path the ghost still has to follow (its future = the forward record further back)."""
    if a <= 0.002:
        return
    i_now = N - k_now                                # ghost is at forward index i_now, moving to lower i
    ahead = int(2.6 / DT)
    i_end = max(0, i_now - ahead)
    if i_now - i_end < 20:
        return
    idx = np.arange(i_now, i_end - 1, -40)
    pts = PF[idx][:, 2:4]
    S = np.empty_like(pts)
    S[:, 0] = PIV[0] + cam.s * pts[:, 0]
    S[:, 1] = PIV[1] - cam.s * pts[:, 1]
    period = 34.0 * cam.z
    chunks = 20
    edges = np.linspace(0, len(S) - 1, chunks + 1).astype(int)
    for i in range(chunks):
        i0, i1 = edges[i], edges[i + 1]
        if i1 <= i0:
            continue
        u = 1 - i / chunks
        p = skia.Path()
        p.moveTo(*S[i0])
        for k in range(i0 + 1, i1 + 1):
            p.lineTo(*S[k])
        # dash phase anchored to the path itself, so the dashes sit still on the track as the ghost runs it
        ph = (cam.s * (ARC[-1] - ARC[idx[i0]])) % period
        pa = paint(WHITE, 0.30 * a * u ** 1.3, stroke=5.0 * cam.z, cap=skia.Paint.kButt_Cap)
        pa.setPathEffect(skia.DashPathEffect.Make([period * 0.55, period * 0.45], ph))
        cv.drawPath(p, pa)


def velocities(Y):
    w = all_omegas(Y[None, :])[0]
    t1, t2 = Y[0], Y[1]
    v1 = np.array([L * w[0] * math.cos(t1), L * w[0] * math.sin(t1)])
    v2 = v1 + np.array([L * w[1] * math.cos(t2), L * w[1] * math.sin(t2)])
    return v1, v2


V1, V2 = velocities(F[N])


def draw_arrow(cv, cam, P, v, f, a):
    """neon velocity arrow on a bob; f = signed scale (+1 forward, -1 reversed, passes 0 in the snap)."""
    if a <= 0.002 or abs(f) < 0.02:
        return
    sp = float(np.hypot(*v))
    Lpx = min(330.0, 130.0 + 24.0 * sp) * f * cam.z
    ux, uy = v[0] / sp, -v[1] / sp
    bx, by = cam.xy(*P)
    off = 30 * cam.z * np.sign(f)
    sx, sy = bx + ux * off, by + uy * off
    ex, ey = bx + ux * Lpx, by + uy * Lpx
    head = 46 * cam.z * min(1.0, abs(f) * 1.5)
    dx, dy = ux * np.sign(f), uy * np.sign(f)
    hx, hy = ex - dx * head, ey - dy * head
    # glow additive; the shaft and head are drawn opaque (src-over) so they read even across a bright arm
    cv.drawLine(sx, sy, hx, hy, paint(PURPLE, 0.20 * a, stroke=34 * cam.z))
    cv.drawLine(sx, sy, hx, hy, paint(PURPLE, 0.95 * a, stroke=13 * cam.z, cap=skia.Paint.kButt_Cap, blend=skia.BlendMode.kSrcOver))
    cv.drawLine(sx, sy, hx, hy, paint(PURPLE_CORE, 0.6 * a, stroke=5 * cam.z, cap=skia.Paint.kButt_Cap))
    tri = skia.Path()
    nx, ny = -dy, dx
    tri.moveTo(ex, ey)
    tri.lineTo(hx + nx * head * 0.62, hy + ny * head * 0.62)
    tri.lineTo(hx - nx * head * 0.62, hy - ny * head * 0.62)
    tri.close()
    cv.drawPath(tri, paint(PURPLE, 0.25 * a, blur=10 * cam.z))
    cv.drawPath(tri, paint(PURPLE, 0.95 * a, blend=skia.BlendMode.kSrcOver))


# ------------------------------------------------------------------ text
def text_w(fnt, s, track=0.0):
    return fnt.measureText(s) + track * max(0, len(s) - 1)


def draw_text(cv, s, x, y, fnt, c, a, align="center", track=0.0, shadow=True):
    if a <= 0.002:
        return
    w = text_w(fnt, s, track)
    x0 = x - w / 2 if align == "center" else (x - w if align == "right" else x)
    if shadow:
        sp = paint((0, 0, 0), 0.75 * a, blend=skia.BlendMode.kSrcOver, blur=14)
        _put(cv, s, x0, y, fnt, sp, track)
    _put(cv, s, x0, y, fnt, paint(c, a, blend=skia.BlendMode.kSrcOver), track)


def _put(cv, s, x, y, fnt, p, track):
    if track == 0:
        cv.drawString(s, x, y, fnt, p)
        return
    for ch in s:
        cv.drawString(ch, x, y, fnt, p)
        x += fnt.measureText(ch) + track


def wrap(s, fnt, maxw):
    words, lines, cur = s.split(" "), [], ""
    for w_ in words:
        tryl = (cur + " " + w_).strip()
        if fnt.measureText(tryl) <= maxw or not cur:
            cur = tryl
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


ACCENT = {"orange": ORANGE, "cyan": CYAN, "purple": PURPLE}


def draw_captions(cv, t):
    for (s, a0, a1, acc) in TL.CAPTIONS:
        if not (a0 - 0.01 <= t <= a1 + 0.25):
            continue
        ain = sstep(a0, a0 + 0.2, t)
        aout = 1 - sstep(a1, a1 + 0.2, t)
        a = ain * aout
        if a <= 0.002:
            continue
        dy = 16 * (1 - ain) - 10 * (1 - aout)
        lines = wrap(s, F_CAP, 1060)
        lh = 106
        y_top = CAP_Y - (len(lines) - 1) * lh / 2
        for i, ln in enumerate(lines):
            draw_text(cv, ln, CX, y_top + i * lh + 33 + dy, F_CAP, WHITE, a)
        if acc:
            yb = y_top + (len(lines) - 1) * lh + 33 + 46 + dy
            c = ACCENT[acc]
            cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(CX - 70, yb, CX + 70, yb + 10), 5, 5), paint(c, 0.35 * a, blur=10))
            cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(CX - 70, yb, CX + 70, yb + 10), 5, 5), paint(c, a, blend=skia.BlendMode.kSrcOver))


def tri(cv, x, y, s, left, c, a):
    p = skia.Path()
    if left:
        p.moveTo(x, y); p.lineTo(x + s, y - s * 0.6); p.lineTo(x + s, y + s * 0.6)
    else:
        p.moveTo(x + s, y); p.lineTo(x, y - s * 0.6); p.lineTo(x, y + s * 0.6)
    p.close()
    cv.drawPath(p, paint(c, a, blend=skia.BlendMode.kSrcOver))


def draw_hud(cv, t):
    xr = SAFE_R - 40                                    # right edge of the HUD (inside the safe zone)
    ph = TL.phase(t)
    if ph in ("hook",):
        simt = 0.0
    elif ph in ("forward", "freeze"):
        simt = TL.sim_forward(t)
    else:
        simt = T_SIM - TL.tau_rev(t)
    simt = max(0.0, simt)
    draw_text(cv, "SIM TIME", xr, 150, F_HUD_L, (150, 160, 175), 0.9, align="right", track=4, shadow=False)
    s = f"{simt:.2f}"
    # fixed-advance digits so the counter doesn't jitter
    adv = F_HUD.measureText("0")
    x = xr - len(s) * adv - F_HUD.measureText(" s")
    for ch in s:
        w_ = F_HUD.measureText(ch)
        cv.drawString(ch, x + (adv - w_) / 2, 212, F_HUD, paint(WHITE, 0.95, blend=skia.BlendMode.kSrcOver))
        x += adv
    cv.drawString(" s", x, 212, F_HUD, paint(WHITE, 0.95, blend=skia.BlendMode.kSrcOver))
    # direction indicator: REVERSED from the flip, back to FORWARD for the loop
    rev = sstep(TL.T_FLIP - 0.04, TL.T_FLIP + 0.04, t) * (1 - sstep(29.25, 29.65, t))
    for (label, left, c, a) in (("FORWARD", False, WHITE, 1 - rev), ("REVERSED", True, PURPLE_CORE, rev)):
        if a <= 0.002:
            continue
        w_ = F_HUD_S.measureText(label)
        draw_text(cv, label, xr, 282, F_HUD_S, c, 0.95 * a, align="right", shadow=False)
        xt = xr - w_ - 22 - 56
        tri(cv, xt, 268, 24, left, c, 0.95 * a)
        tri(cv, xt + 26, 268, 24, left, c, 0.95 * a)
    sp = TL.speed(t)
    draw_text(cv, f"×{sp:.1f}", xr, 344, F_HUD_S, (190, 200, 215), 0.9, align="right", shadow=False)


def draw_tag(cv):
    draw_text(cv, TAG, 64, 196, F_TAG, CYAN, 0.95, align="left", track=6, shadow=False)


# ------------------------------------------------------------------ graph
GT = np.arange(0, N + 1, 200)                         # every 20 ms of sim time
G_TAU = GT * DT
G_C = np.log10(np.maximum(ANG_C[GT], 1e-16))
G_O = np.log10(np.maximum(ANG_O[GT], 1e-16))
YMIN, YMAX = -16.6, 1.0


def draw_graph(cv, t):
    a = sstep(TL.T_GRAPH, TL.T_GRAPH + 0.3, t) * (1 - sstep(TL.T_GRAPH_OUT, TL.T_GRAPH_OUT + 0.3, t))
    if a <= 0.002:
        return
    x0, y0, x1, y1 = GRAPH
    sc = 0.96 + 0.04 * sstep(TL.T_GRAPH, TL.T_GRAPH + 0.35, t)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cv.save()
    cv.translate(cx, cy); cv.scale(sc, sc); cv.translate(-cx, -cy)
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x0, y0, x1, y1), 26, 26)
    cv.drawRRect(rr, paint((0, 0, 0), 0.84 * a, blend=skia.BlendMode.kSrcOver))
    cv.drawRRect(rr, paint(WHITE, 0.16 * a, stroke=2.5, blend=skia.BlendMode.kSrcOver))
    px0, px1, py0, py1 = x0 + 150, x1 - 44, y0 + 96, y1 - 112    # plot area
    draw_text(cv, "Separation from the perfect rewind (rad)", x0 + 40, y0 + 60, F_G, (225, 230, 240), a, align="left", shadow=False)

    def X(tau):
        return px0 + (px1 - px0) * tau / T_SIM

    def Y(lg):
        return py1 - (py1 - py0) * (lg - YMIN) / (YMAX - YMIN)
    # grid + ticks
    for e in (-16, -12, -8, -4, 0):
        yy = Y(e)
        cv.drawLine(px0, yy, px1, yy, paint(WHITE, 0.08 * a, stroke=2, blend=skia.BlendMode.kSrcOver))
        sup = str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        draw_text(cv, "10" + sup, px0 - 18, yy + 11, F_G_S, (175, 185, 200), a, align="right", shadow=False)
    for s in (0, 5, 10, 15, 20):
        xx = X(s)
        cv.drawLine(xx, py1, xx, py1 + 10, paint(WHITE, 0.5 * a, stroke=2.5, blend=skia.BlendMode.kSrcOver))
        draw_text(cv, str(s), xx, py1 + 44, F_G_S, (175, 185, 200), a, shadow=False)
    cv.drawLine(px0, py0 - 8, px0, py1, paint(WHITE, 0.55 * a, stroke=2.5, blend=skia.BlendMode.kSrcOver))
    cv.drawLine(px0, py1, px1, py1, paint(WHITE, 0.55 * a, stroke=2.5, blend=skia.BlendMode.kSrcOver))
    draw_text(cv, "seconds after the reversal", (px0 + px1) / 2, py1 + 88, F_G_S, (140, 150, 165), a, shadow=False)
    # data, drawn on as the graph opens, then following the live rewind
    tau_now = TL.tau_rev(t)
    reveal = sstep(TL.T_GRAPH + 0.05, TL.T_GRAPH + 1.0, t)
    tmax = tau_now * reveal
    nshow = int(np.searchsorted(G_TAU, tmax))
    cv.save()
    cv.clipRect(skia.Rect.MakeLTRB(px0, py0 - 30, px1 + 6, py1))
    for (G, c, core) in ((G_C, CYAN, CYAN_CORE), (G_O, ORANGE, ORANGE_CORE)):
        if nshow < 2:
            continue
        p = skia.Path()
        p.moveTo(X(G_TAU[0]), Y(G[0]))
        for k in range(1, nshow):
            p.lineTo(X(G_TAU[k]), Y(G[k]))
        cv.drawPath(p, paint(c, 0.22 * a, stroke=16))
        cv.drawPath(p, paint(c, 0.95 * a, stroke=5.5))
        cv.drawCircle(X(G_TAU[nshow - 1]), Y(G[nshow - 1]), 9, paint(core, a))
    # fitted exponential (straight line on a log axis), common window, appears after the data
    fa = a * sstep(TL.T_GRAPH + 1.0, TL.T_GRAPH + 1.5, t)
    if fa > 0.002:
        for key, c in (("common_cyan", CYAN_CORE), ("common_orange", ORANGE_CORE)):
            f_ = FIT[key]
            ta, tb = 0.0, 13.5 if key == "common_orange" else 21.0
            la = (f_["intercept"] + f_["slope"] * ta) / math.log(10)
            lb = (f_["intercept"] + f_["slope"] * tb) / math.log(10)
            pa = paint(c, 0.7 * fa, stroke=3, cap=skia.Paint.kButt_Cap, blend=skia.BlendMode.kSrcOver)
            pa.setPathEffect(skia.DashPathEffect.Make([14, 12], 0))
            cv.drawLine(X(ta), Y(la), X(tb), Y(lb), pa)
    cv.restore()
    la = a * min(1.0, reveal * 3)
    for i, (lab, c) in enumerate((("nudge 10⁻¹⁰°", ORANGE), ("rounding only", CYAN))):
        yy = Y(-8.6) + i * 48
        lx = X(12.4)
        cv.drawLine(lx, yy - 11, lx + 42, yy - 11, paint(c, la, stroke=6, blend=skia.BlendMode.kSrcOver))
        draw_text(cv, lab, lx + 56, yy, F_G_S, (225, 230, 240), la, align="left", shadow=False)
    lam = FIT["lambda_mean_common"]
    draw_text(cv, f"slope λ ≈ {lam:.2f} s⁻¹", X(12.4), Y(-14.4), F_G, WHITE, fa, align="left", shadow=False)
    cv.restore()


# ------------------------------------------------------------------ bloom
SMALL = (W // 4, H // 4)
_bl = {}


def bloom(img):
    if not _bl:
        _bl["s"] = skia.Surface(*SMALL)
        _bl["b1"] = skia.Surface(*SMALL)
        _bl["b2"] = skia.Surface(*SMALL)
    s = _bl["s"]
    sc = s.getCanvas()
    sc.clear(skia.ColorBLACK)
    sc.drawImageRect(img.withDefaultMipmaps(), skia.Rect.MakeWH(*SMALL),
                     skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear))
    small = s.makeImageSnapshot()
    out = []
    for key, sig in (("b1", 3.5), ("b2", 11.0)):
        b = _bl[key]
        bc = b.getCanvas()
        bc.clear(skia.ColorBLACK)
        p = skia.Paint()
        p.setImageFilter(skia.ImageFilters.Blur(sig, sig, skia.TileMode.kDecal))
        bc.drawImage(small, 0, 0, skia.SamplingOptions(), p)
        out.append(b.makeImageSnapshot())
    return out


# ------------------------------------------------------------------ frame
def frame(t, surf=None, emit=None):
    if surf is None:
        surf = skia.Surface(W, H)
        emit = skia.Surface(W, H)
    cam = Cam(t)
    ph = TL.phase(t)
    ec = emit.getCanvas()
    ec.clear(skia.ColorBLACK)
    draw_pivot(ec, cam)

    # which state each copy shows
    if ph == "hook":
        kf = 0
    elif ph in ("forward", "freeze"):
        kf = int(round(TL.sim_forward(t) / DT))
    tau = TL.tau_rev(t)
    kr = min(N, int(round(tau / DT)))
    reverse_on = t >= TL.T_REWIND
    if not reverse_on:
        Pg = Pc = Po = PF[kf]
    else:
        Pg, Pc, Po = PF[N - kr], PC[kr], PO[kr]

    # alphas
    ignite = sstep(TL.T_RELEASE, TL.T_RELEASE + 0.25, t)                 # copies switch on at release
    out_ = 1 - sstep(*TL.T_FADE_COPIES, t)                              # and fade at the end
    a_col = ignite * out_
    breathe = (1 - sstep(1.75, 2.0, t)) + sstep(28.6, 29.2, t)          # gentle 'held' pulse in hook/landed
    pulse = 1.0 + 0.10 * breathe * math.cos(2 * math.pi * t / 1.5)      # 30 s = 20 periods: loop-continuous
    a_ghost = pulse

    # trails
    if ph in ("forward", "freeze") and t >= TL.T_RELEASE:
        fade = 1 - sstep(9.3, 10.8, t)
        k0 = max(0, kf - int(2.2 / DT))
        idx = np.arange(k0, kf + 1, 60)
        if len(idx) > 2:
            draw_trail(ec, cam, PF[idx][:, 2:4], CYAN, 7.0, 0.85 * fade * a_col)
            draw_trail(ec, cam, PF[idx][:, 2:4], ORANGE, 7.0, 0.85 * fade * a_col)
    if reverse_on:
        draw_ghost_path(ec, cam, kr, sstep(11.0, 11.6, t))
        k0 = max(0, kr - int(1.3 / DT))
        idx = np.arange(k0, kr + 1, 40)
        if len(idx) > 2:
            draw_trail(ec, cam, PC[idx][:, 2:4], CYAN, 5.5, 0.9 * out_)
            draw_trail(ec, cam, PO[idx][:, 2:4], ORANGE, 5.5, 0.9 * out_)

    draw_copy(ec, cam, Pg, "ghost", a_ghost)
    draw_copy(ec, cam, Pc, "cyan", a_col)
    draw_copy(ec, cam, Po, "orange", a_col)

    # velocity arrows: appear on the freeze, snap round at the flip, fade as the rewind starts
    if 9.0 <= t <= 11.6:
        a_ar = sstep(9.15, 9.45, t) * (1 - sstep(11.0, 11.45, t))
        u = (t - TL.T_FLIP) / 0.16
        if u <= 0:
            f = 1.0
        elif u >= 1:
            f = -1.0 - 0.18 * math.exp(-(t - TL.T_FLIP - 0.16) * 14) * math.cos((t - TL.T_FLIP - 0.16) * 40)
        else:
            f = math.cos(math.pi * u * u)
        Pa = PF[N]
        draw_arrow(ec, cam, Pa[0:2], V1, f, a_ar)
        draw_arrow(ec, cam, Pa[2:4], V2, f, a_ar)
        fl = math.exp(-max(0.0, t - TL.T_FLIP - 0.08) * 9) if t >= TL.T_FLIP + 0.08 else 0.0
        if fl > 0.01:
            for (bx, by) in (cam.xy(*Pa[0:2]), cam.xy(*Pa[2:4])):
                r = 40 + 160 * (1 - fl)
                ec.drawCircle(bx, by, r, paint(PURPLE, 0.8 * fl, stroke=8))

    img = emit.makeImageSnapshot()
    b1, b2 = bloom(img)
    cv = surf.getCanvas()
    cv.clear(skia.ColorBLACK)
    cv.drawImage(img, 0, 0)
    samp = skia.SamplingOptions(skia.FilterMode.kLinear)
    for b, g in ((b1, 0.85), (b2, 0.75)):
        p = skia.Paint(); p.setBlendMode(skia.BlendMode.kPlus); p.setAlphaf(g)
        cv.drawImageRect(b, skia.Rect.MakeWH(W, H), samp, p)

    draw_graph(cv, t)
    draw_captions(cv, t)
    draw_hud(cv, t)
    draw_tag(cv)
    return surf


def rgb(surf):
    a = surf.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
    return np.ascontiguousarray(a[:, :, :3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs=2)
    ap.add_argument("--seg", nargs=3)
    args = ap.parse_args()
    if args.still:
        t = float(args.still[0])
        s = frame(t)
        s.makeImageSnapshot().save(args.still[1], skia.kPNG)
        return
    f0, f1, out = int(args.seg[0]), int(args.seg[1]), args.seg[2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "ffv1", "-level", "3", "-threads", "2", out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    surf, emit = skia.Surface(W, H), skia.Surface(W, H)
    import time
    t0 = time.time()
    for fi in range(f0, f1):
        t = fi / FPS
        frame(t, surf, emit)
        pr.stdin.write(rgb(surf).tobytes())
        if (fi - f0) % 60 == 0:
            print(f"[{out}] frame {fi} ({(fi - f0 + 1) / (time.time() - t0):.1f} fps)", flush=True)
    pr.stdin.close()
    pr.wait()


if __name__ == "__main__":
    main()
