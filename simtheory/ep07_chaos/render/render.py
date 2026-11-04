"""Stage 3: deterministic offline frame renderer.

Reads the saved simulation states (data/run_*.npy) and the measurements, and draws every frame
from the video-time -> sim-time mapping in timeline.py. No real-time capture.

  python render/render.py frames  [start end step]   -> out/frames/f_XXXX.png
  python render/render.py still   t1 t2 ...          -> out/stills/still_<t>.png
"""
import math
import os
import sys
from functools import lru_cache

import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import timeline as T  # noqa: E402

W, H = T.W, T.H
OUT = os.path.join(ROOT, "out")
FONT_DIR = "/usr/share/fonts/opentype/inter"
F_BOLD = os.path.join(FONT_DIR, "Inter-Bold.otf")
F_REG = os.path.join(FONT_DIR, "Inter-Regular.otf")
F_MED = os.path.join(FONT_DIR, "Inter-Medium.otf")
F_LIGHT = os.path.join(FONT_DIR, "Inter-Light.otf")


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


C = {k: hexrgb(v) for k, v in dict(cyan="#22E5FF", blue="#3D7BFF", purple="#9B5CFF", magenta="#FF4FD8",
                                    orange="#FF8A2A", white="#FFFFFF", gold="#FFC94A", grey="#9AA3B5").items()}

# ---------------------------------------------------------------- sim data
DATA = T.DATA
RUNS = {k: np.load(os.path.join(DATA, f"run_{k}.npy"))[:, :2] for k in ["f32", "f64", "mp100"]}
DT = T.M["dt"]
COPIES = [("f32", "orange"), ("f64", "cyan"), ("mp100", "white")]


def angles(name, tau):
    """theta1, theta2 at sim time tau (linear interpolation between the saved 1 ms states)."""
    a = RUNS[name]
    x = np.clip(np.asarray(tau, float) / DT, 0, len(a) - 1.000001)
    i = np.floor(x).astype(int)
    f = (x - i)[..., None]
    return a[i] * (1 - f) + a[i + 1] * f


def wrapdeg(d):
    d = np.degrees(d)
    return np.abs((d + 180) % 360 - 180)


# ---------------------------------------------------------------- layout
L_PX = 280.0                       # 1 m arm at zoom 1 (full reach 560 px keeps the pendulum in frame)
PIVOT_FULL = (T.SAFE_CX, T.SAFE_CY)
# chart plot box (digits 0..45 on x, seconds 0..70 on y)
X0, X1, YB, YT = 250.0, 1110.0, 1440.0, 900.0
DX_MAX, DY_MAX = 45.0, 70.0
PIVOT_INSET = (960.0, 1290.0)
L_INSET = 57.5


def cx(d):
    return X0 + d / DX_MAX * (X1 - X0)


def cy(s):
    return YB - s / DY_MAX * (YB - YT)


def view(t):
    """pivot (px), px per metre, pendulum intensity."""
    m = T.chart_mix(t)
    z = T.zoom(t)
    k_full = L_PX * z
    px = PIVOT_FULL[0] + (PIVOT_INSET[0] - PIVOT_FULL[0]) * m
    py = PIVOT_FULL[1] + (PIVOT_INSET[1] - PIVOT_FULL[1]) * m
    k = math.exp(math.log(k_full) + (math.log(L_INSET) - math.log(k_full)) * m)
    dim = 1 - 0.55 * m
    return (px, py), k, dim


def pulse(t):
    h = T.hold_phase(t)
    if h is None:
        return 1.0
    return 1.0 + 0.10 * math.sin(math.pi * h / 0.8) ** 2


def positions(th, pivot, k):
    t1, t2 = th[..., 0], th[..., 1]
    x1 = pivot[0] + k * np.sin(t1)
    y1 = pivot[1] + k * np.cos(t1)
    x2 = x1 + k * np.sin(t2)
    y2 = y1 + k * np.cos(t2)
    return x1, y1, x2, y2


# ---------------------------------------------------------------- cairo helpers
def a8():
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    c.set_operator(cairo.OPERATOR_ADD)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    return s, c


def a8_to_float(s):
    s.flush()
    buf = np.ndarray((H, s.get_stride()), np.uint8, s.get_data())[:, :W]
    return buf.astype(np.float32) * (1 / 255)


def draw_pendulum_copy(c, name, t, pivot, k, dim):
    """motion-blurred rods + bobs + 1 s trail into an A8 coverage layer."""
    tau = T.sim_time(t)
    tau_prev = T.sim_time(t - 1 / T.FPS) if T.loop_time(t) > 0 else tau
    # 180-degree shutter: samples over the second half of the frame interval
    taus_probe = np.linspace(tau - 0.5 * (tau - tau_prev), tau, 12)
    x1, y1, x2, y2 = positions(angles(name, taus_probe), pivot, k)
    arc = np.sum(np.hypot(np.diff(x2), np.diff(y2)))
    K = int(np.clip(math.ceil(arc / 3.0), 1, 40))
    taus = np.linspace(tau - 0.5 * (tau - tau_prev), tau, K) if K > 1 else np.array([tau])
    x1, y1, x2, y2 = positions(angles(name, taus), pivot, k)
    s = k / L_PX
    rod_w = max(3.2, 6.0 * s)
    bob_r = max(6.0, 13.0 * s)
    a = 1.0 / K

    # trail: where the lower bob was over the last ~1 s of video (also correct while rewinding)
    trail_w = T.clamp(1.0 - T.smooth(T.REWIND_T0, T.REWIND_T0 + 1.6, T.loop_time(t))) if T.rewinding(t) or T.loop_time(t) >= T.REWIND_T1 else 1.0
    trail_fade = 1.0 - T.smooth(T.REWIND_T0 + 0.6, T.REWIND_T1 - 0.4, T.loop_time(t)) if T.loop_time(t) > T.REWIND_T0 else 1.0
    if trail_w > 0.01 and trail_fade > 0.01:
        nu = 240
        us = np.linspace(0, 1.0 * trail_w, nu)
        ttau = np.array([T.sim_time(t - u) for u in us])
        if np.ptp(ttau) > 1e-6:
            span = np.ptp(ttau)
            n = int(np.clip(span / 0.004, 24, 1500))
            us2 = np.linspace(0, 1.0 * trail_w, n)
            ttau = np.interp(us2, us, ttau)
            # never longer than the normal 1 s trail (1.3 s of sim); shrinks with trail_w while rewinding
            keep = np.abs(ttau - tau) <= T.SPEED * 1.0 * trail_w + 1e-6
            ttau = ttau[: max(2, int(np.argmin(keep)) if not keep.all() else len(keep))]
            n = len(ttau)
            _, _, tx, ty = positions(angles(name, ttau), pivot, k)
            chunks = 30
            idx = np.linspace(0, n - 1, chunks + 1).astype(int)
            c.set_line_cap(cairo.LINE_CAP_BUTT)
            c.set_line_join(cairo.LINE_JOIN_ROUND)
            for j in range(chunks):
                i0, i1 = idx[j], idx[j + 1]
                if i1 <= i0:
                    continue
                f = 1 - j / chunks                     # 1 at the head, 0 at the tail
                c.set_line_width(max(3.0, 4.5 * s * (0.5 + 0.5 * f)))
                c.set_source_rgba(0, 0, 0, 0.55 * f * f * trail_fade)
                c.move_to(tx[i0], ty[i0])
                for i in range(i0 + 1, i1 + 1):
                    c.line_to(tx[i], ty[i])
                c.stroke()

    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_width(rod_w)
    for i in range(K):
        c.set_source_rgba(0, 0, 0, 0.85 * a)
        c.move_to(pivot[0], pivot[1])
        c.line_to(x1[i], y1[i])
        c.line_to(x2[i], y2[i])
        c.stroke()
        c.set_source_rgba(0, 0, 0, a)
        c.arc(x1[i], y1[i], bob_r * 0.85, 0, 2 * math.pi)
        c.fill()
        c.arc(x2[i], y2[i], bob_r, 0, 2 * math.pi)
        c.fill()


def split_ring(c, name, tsplit, hsplit, t, pivot, k):
    lt = T.loop_time(t)
    u = (lt - tsplit) / 0.75
    if not (0 <= u <= 1):
        return
    th = angles(name, hsplit)
    _, _, x2, y2 = positions(th, pivot, k)
    r = 22 + 150 * (1 - (1 - u) ** 3)
    c.set_line_width(max(3.0, 7 * (1 - u)))
    c.set_source_rgba(0, 0, 0, 0.9 * (1 - u) ** 1.5)
    c.arc(float(x2), float(y2), r, 0, 2 * math.pi)
    c.stroke()


# ---------------------------------------------------------------- chart (glowing marks go through bloom)
def chart_glow(t, layers):
    """Draw chart marks into per-colour coverage layers. Returns nothing; layers = dict colour -> (surf, ctx)."""
    lt = T.loop_time(t)
    ca = T.chart_alpha(t)
    if ca <= 0:
        return

    def L(col):
        if col not in layers:
            layers[col] = a8()
        return layers[col][1]

    # axes draw in from the origin
    g = T.smooth(21.3, 22.1, lt)
    c = L("axis")
    c.set_line_width(3)
    c.set_line_cap(cairo.LINE_CAP_SQUARE)
    c.set_source_rgba(0, 0, 0, ca)
    c.move_to(X0, YB); c.line_to(X0 + (X1 - X0) * g, YB); c.stroke()
    c.move_to(X0, YB); c.line_to(X0, YB - (YB - YT) * g); c.stroke()
    ta = ca * T.smooth(21.8, 22.3, lt)
    c.set_source_rgba(0, 0, 0, ta)
    for d in (10, 20, 30, 40):
        c.move_to(cx(d), YB); c.line_to(cx(d), YB + 14); c.stroke()
    for s in (20, 40, 60):
        c.move_to(X0, cy(s)); c.line_to(X0 - 14, cy(s)); c.stroke()

    # magenta "real measurements" zone: fewer than 6 digits (left of every simulated point)
    ra = ca * T.smooth(T.REAL_T, T.REAL_T + 0.5, lt)
    if ra > 0:
        c = L("magenta_fill")
        c.set_source_rgba(0, 0, 0, 0.16 * ra)
        c.rectangle(X0 + 2, YT, cx(6) - X0 - 2, YB - YT - 2)
        c.fill()
        c = L("magenta")
        c.set_line_width(3)
        c.set_source_rgba(0, 0, 0, 0.9 * ra)
        c.set_dash([14, 10])
        c.move_to(cx(6), YB); c.line_to(cx(6), YT); c.stroke()
        c.set_dash([])

    # gold 60-second line
    ma = ca * T.smooth(T.MINUTE_T, T.MINUTE_T + 0.6, lt)
    if ma > 0:
        c = L("gold")
        c.set_line_width(3)
        c.set_source_rgba(0, 0, 0, 0.85 * ma)
        c.set_dash([18, 12])
        gx = T.smooth(T.MINUTE_T, T.MINUTE_T + 0.6, lt)
        c.move_to(X0, cy(60)); c.line_to(X0 + (X1 - X0) * gx, cy(60)); c.stroke()
        c.set_dash([])
        u = T.smooth(T.MINUTE_T + 0.3, T.MINUTE_T + 0.8, lt)
        if u > 0:
            c.set_line_width(4)
            c.set_source_rgba(0, 0, 0, ma * u)
            c.arc(cx(T.D60), cy(60), 26 + 8 * (1 - u), 0, 2 * math.pi)
            c.stroke()

    # fitted straight line, drawn left to right
    fa = T.smooth(T.FIT_T0, T.FIT_T1, lt)
    if fa > 0:
        c = L("blue")
        c.set_line_width(4)
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        c.set_source_rgba(0, 0, 0, 0.95 * ca)
        d0, d1 = 3.0, 45.0
        de = d0 + (d1 - d0) * fa
        c.move_to(cx(d0), cy(T.INTERCEPT + T.SLOPE * d0))
        c.line_to(cx(de), cy(T.INTERCEPT + T.SLOPE * de))
        c.stroke()

    # data points pop in one by one
    for (tp, (d, hz, name)) in zip(T.CHART_POINT_TIMES, T.CHART_POINTS):
        u = (lt - tp) / 0.30
        if u < 0:
            continue
        col = {"f32": "orange", "f64": "cyan"}.get(name, "purple")
        c = L(col)
        sc = 1.0 + 0.35 * math.sin(math.pi * min(u, 1.0)) if u < 1 else 1.0
        c.set_source_rgba(0, 0, 0, ca)
        c.arc(cx(d), cy(hz), 14 * sc, 0, 2 * math.pi)
        c.fill()
        ur = (lt - tp) / 0.6
        if 0 <= ur <= 1:
            c.set_line_width(max(3.0, 6 * (1 - ur)))
            c.set_source_rgba(0, 0, 0, ca * (1 - ur))
            c.arc(cx(d), cy(hz), 18 + 60 * ur, 0, 2 * math.pi)
            c.stroke()


LAYER_COL = {"orange": C["orange"], "cyan": C["cyan"], "white": C["white"], "purple": C["purple"],
             "blue": C["blue"], "gold": C["gold"], "magenta": C["magenta"], "magenta_fill": C["magenta"],
             "axis": (0.62, 0.66, 0.74)}
LAYER_GAIN = {"magenta_fill": 0.7, "axis": 0.85}


# ---------------------------------------------------------------- bloom + tone map
def bloom(hdr):
    small = cv2.resize(hdr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for sig, wgt in ((2.5, 0.55), (7.0, 0.40), (18.0, 0.30), (40.0, 0.18)):
        acc += wgt * cv2.GaussianBlur(small, (0, 0), sig)
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def tonemap(x):
    knee = 0.72
    y = np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))
    return np.clip(y, 0, 1)


# ---------------------------------------------------------------- text
@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=512)
def text_sprite(text, path, size, rgb, shadow, tnum=False):
    """RGBA float sprite (premultiplied) + anchor offsets. Shadow is a soft dark halo."""
    f = font(path, size)
    feats = ["tnum"] if tnum else None
    l, t_, r, b = f.getbbox(text, features=feats)
    pad = int(size * 0.6) if shadow else 4
    w, h = r - l + 2 * pad, b - t_ + 2 * pad
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t_), text, font=f, fill=255, features=feats)
    a = np.asarray(m, np.float32) / 255
    rgba = np.zeros((h, w, 4), np.float32)
    if shadow:
        sh = cv2.GaussianBlur(a, (0, 0), size * 0.16)
        sh = np.clip(sh * 1.6, 0, 1) * 0.78
        # composite text over shadow (premultiplied)
        alpha = a + sh * (1 - a)
        rgba[..., :3] = a[..., None] * np.array(rgb, np.float32)
        rgba[..., 3] = alpha
    else:
        rgba[..., :3] = a[..., None] * np.array(rgb, np.float32)
        rgba[..., 3] = a
    # ink bounds relative to sprite origin (for layout + safe-area checks)
    return rgba, pad, (r - l, b - t_), (l, t_)


class Canvas:
    def __init__(self, img):
        self.img = img
        self.boxes = []

    def text(self, text, x, y, size, path=F_BOLD, rgb=(1, 1, 1), alpha=1.0, align="left", shadow=True,
             tnum=False, rotate=False, check=True):
        """(x, y) = top of the ink box; align left/center/right on x."""
        if alpha <= 0.003:
            return None
        spr, pad, (iw, ih), _ = text_sprite(text, path, size, tuple(rgb), shadow, tnum)
        if rotate:
            spr = np.ascontiguousarray(np.rot90(spr, 1))
            iw, ih = ih, iw
        if align == "center":
            x0 = x - iw / 2
        elif align == "right":
            x0 = x - iw
        else:
            x0 = x
        if rotate:   # for rotated text, (x, y) is the centre
            x0, y0 = x - iw / 2, y - ih / 2
        else:
            y0 = y
        if check:
            self.boxes.append((text, x0, y0, x0 + iw, y0 + ih, alpha))
        ox, oy = int(round(x0)) - pad, int(round(y0)) - pad
        self.blit(spr, ox, oy, alpha)
        return (x0, y0, x0 + iw, y0 + ih)

    def blit(self, spr, ox, oy, alpha):
        h, w = spr.shape[:2]
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + w, W), min(oy + h, H)
        if x1 <= x0 or y1 <= y0:
            return
        s = spr[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        dst = self.img[y0:y1, x0:x1]
        dst *= (1 - s[..., 3:4] * alpha)
        dst += s[..., :3] * alpha


def mix(c1, c2, f):
    return tuple(a + (b - a) * f for a, b in zip(c1, c2))


def fmt_err(e):
    if e < 1:
        return f"{e:.6f}°"
    if e < 10:
        return f"{e:.2f}°"
    return f"{e:.1f}°"


def draw_shapes_rewind(img, x_right, y_mid, alpha):
    """two left-pointing triangles drawn as shapes (no glyphs)"""
    if alpha <= 0:
        return
    s = cairo.ImageSurface(cairo.FORMAT_A8, 80, 50)
    c = cairo.Context(s)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    for ox in (0, 30):
        c.move_to(ox + 2, 25); c.line_to(ox + 30, 8); c.line_to(ox + 30, 42); c.close_path()
    c.fill()
    s.flush()
    a = np.ndarray((50, s.get_stride()), np.uint8, s.get_data())[:, :80].astype(np.float32) / 255
    spr = np.zeros((50, 80, 4), np.float32)
    spr[..., :3] = a[..., None] * np.array(C["magenta"], np.float32)
    spr[..., 3] = a
    Canvas(img).blit(spr, int(x_right - 80), int(y_mid - 25), alpha)


# ---------------------------------------------------------------- frame
def render(t, cover=False):
    t = T.loop_time(t)          # the last frame is rendered as frame 0 (seamless loop)
    lt = t
    pivot, k, dim = view(t)
    p = pulse(t)

    layers = {}
    # pendulum copies
    for name, col in COPIES:
        layers[col] = a8()
        draw_pendulum_copy(layers[col][1], name, t, pivot, k, dim)
    c = layers["white"][1]
    c.set_source_rgba(0, 0, 0, 0.8)
    c.arc(pivot[0], pivot[1], max(5.0, 9.0 * k / L_PX), 0, 2 * math.pi)
    c.fill()
    # split bursts
    split_ring(layers["orange"][1], "f32", T.T7, T.H7, t, pivot, k)
    split_ring(layers["cyan"][1], "f64", T.T16, T.H16, t, pivot, k)

    pend_gain = {"orange": 0.95 * T.orange_focus(t), "cyan": 0.95, "white": 0.95}
    chart_glow(t, layers)

    hdr = np.zeros((H, W, 3), np.float32)
    for col, (s, _) in layers.items():
        g = pend_gain.get(col, None)
        if g is not None:
            g = g * dim * p
        else:
            g = LAYER_GAIN.get(col, 1.0)
        cov = a8_to_float(s)
        hdr += cov[..., None] * (np.array(LAYER_COL[col], np.float32) * g)
    img = tonemap(hdr + 0.85 * bloom(hdr))
    img = np.ascontiguousarray(img, dtype=np.float32)
    cv = Canvas(img)

    # ---------------- brand + HUD (always)
    cv.text("SIM #07", 100, 400, 44, F_LIGHT, C["cyan"], 1.0, shadow=True)
    tau = T.sim_time(t)
    hud_r = 1146
    cv.text(f"{tau:5.2f} s", hud_r, 400, 44, F_MED, C["white"], 1.0, align="right", tnum=True)
    cv.text("SIM TIME", hud_r - 196, 400, 44, F_REG, C["grey"], 1.0, align="right")
    if T.rewinding(t):
        cv.text("REWIND", hud_r, 458, 44, F_MED, C["magenta"], 1.0, align="right")
        draw_shapes_rewind(img, hud_r - 186, 480, 1.0)
    else:
        cv.text(T.hud_speed_text(t), hud_r, 458, 44, F_MED, C["white"], 1.0, align="right")
        cv.text("SPEED", hud_r - 92, 458, 44, F_REG, C["grey"], 1.0, align="right")

    # ---------------- hook
    ha = 1.0 if cover else T.hook_alpha(t)
    if ha > 0:
        # left-aligned so the "?" clears the up-right pointing arm on frame 1
        cv.text(T.HOOK_LINES[0], 100, 524, 126, F_BOLD, C["white"], ha)
        cv.text(T.HOOK_LINES[1], 100, 524 + 144, 126, F_BOLD, C["white"], ha)

    # ---------------- captions (upper-middle of the safe area)
    for a, b, lines in ([] if cover else T.CAPTIONS):
        ca = T.fade_window(lt, a, b, 0.2)
        if ca <= 0:
            continue
        rise = 14 * (1 - T.smooth(a, a + 0.25, lt))
        for i, line in enumerate(lines):
            cv.text(line, T.SAFE_CX, 540 + i * 116 + rise, 96, F_BOLD, C["white"], ca, align="center")

    # ---------------- legend (bottom-left)
    la = T.legend_alpha(t)
    if la > 0:
        rows = [("f32", "orange", "7 digits"), ("f64", "cyan", "16 digits"), ("mp100", "white", "100 digits (reference)")]
        ref = angles("mp100", tau)
        for i, (name, col, label) in enumerate(rows):
            y = 1500 + i * 56
            rgb = C[col]
            dot = np.zeros((40, 40, 4), np.float32)
            yy, xx = np.mgrid[0:40, 0:40]
            d = np.hypot(xx - 19.5, yy - 19.5)
            core = np.clip(11 - d, 0, 1)
            halo = np.exp(-(d / 9) ** 2) * 0.6
            al = np.clip(core + halo * (1 - core), 0, 1)
            dot[..., :3] = al[..., None] * np.array(rgb, np.float32)
            dot[..., 3] = al
            cv.blit(dot, 96, int(y + 2), la)
            cv.text(label, 140, y, 44, F_REG, mix(rgb, (1, 1, 1), 0.25), la)
            if name != "mp100":
                ra = T.readout_alpha(t, name)
                if ra > 0:
                    e = float(wrapdeg(angles(name, tau)[1] - ref[1]))
                    cv.text("off by", 372, y, 44, F_REG, C["grey"], ra)
                    cv.text(fmt_err(e), 500, y, 44, F_MED, mix(rgb, (1, 1, 1), 0.25), ra, tnum=True)

    # ---------------- chart labels
    ca = T.chart_alpha(t)
    if ca > 0:
        ta = ca * T.smooth(21.4, 21.9, lt)
        cv.text("Prediction time vs digits kept", T.SAFE_CX, 806, 52, F_BOLD, C["white"], ta, align="center")
        la2 = ca * T.smooth(21.8, 22.3, lt)
        for d in (0, 10, 20, 30, 40):
            cv.text(str(d), cx(d), YB + 24, 44, F_REG, (0.85, 0.87, 0.92), la2, align="center")
        for s in (0, 20, 40, 60):
            cv.text(str(s), X0 - 22, cy(s) - 17, 44, F_REG, (0.85, 0.87, 0.92), la2, align="right")
        cv.text("digits kept", (X0 + X1) / 2, YB + 96, 44, F_REG, C["white"], la2, align="center")
        cv.text("seconds of correct prediction", 128, (YB + YT) / 2, 44, F_REG, C["white"], la2, rotate=True)
        for tp, (d, hz, name) in zip(T.CHART_POINT_TIMES, T.CHART_POINTS):
            pa = ca * T.smooth(tp + 0.05, tp + 0.35, lt)
            if pa <= 0:
                continue
            col = {"f32": "orange", "f64": "cyan"}.get(name, "purple")
            rgb = mix(C[col], (1, 1, 1), 0.35)
            lab = f"{hz:.1f} s"
            px, py = cx(d), cy(hz)
            if name == "mp40":
                cv.text(lab, px - 24, py - 66, 44, F_MED, rgb, pa, align="right")
            else:
                cv.text(lab, px + 26, py + 10, 44, F_MED, rgb, pa)
        sa = ca * T.smooth(T.SLOPE_LABEL_T, T.SLOPE_LABEL_T + 0.4, lt)
        if sa > 0:
            cv.text(f"+{T.SLOPE:.1f} s per digit", 700, 1102, 44, F_MED, mix(C["blue"], (1, 1, 1), 0.45), sa,
                    align="right")
        ma = ca * T.smooth(T.MINUTE_T + 0.2, T.MINUTE_T + 0.7, lt)
        if ma > 0:
            cv.text("1 minute", X0 + 18, cy(60) + 14, 44, F_MED, C["gold"], ma)
        ra = ca * T.smooth(T.REAL_T + 0.2, T.REAL_T + 0.7, lt)
        if ra > 0:
            cv.text("real measurements", X0 + 18, YT + 8, 44, F_MED, mix(C["magenta"], (1, 1, 1), 0.25), ra)

    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out, cv.boxes


def safe_violations(boxes):
    x0s, y0s, x1s, y1s = T.SAFE
    bad = []
    for text, a, b, c, d, al in boxes:
        if al > 0.02 and (a < x0s - 0.5 or b < y0s - 0.5 or c > x1s + 0.5 or d > y1s + 0.5):
            bad.append((text, round(a), round(b), round(c), round(d)))
    return bad


def save_png(arr, path, level=1):
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, level])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "frames":
        a, b, st = (int(x) for x in sys.argv[2:5])
        os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
        viol = []
        for f in range(a, b, st):
            arr, boxes = render(f / T.FPS)
            v = safe_violations(boxes)
            if v:
                viol.append((f, v))
            save_png(arr, os.path.join(OUT, "frames", f"f_{f:04d}.png"))
        if viol:
            with open(os.path.join(OUT, "frames", f"safe_violations_{a}.txt"), "w") as fh:
                for f, v in viol:
                    fh.write(f"{f} {v}\n")
    elif mode == "cover":
        # cover image: the moment the orange copy splits away, with the hook text, 1080x1920
        t = float(sys.argv[2])
        arr, boxes = render(t, cover=True)
        print("safe-area violations:", safe_violations(boxes))
        small = cv2.resize(arr, (1080, 1920), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.join(OUT, "delivery"), exist_ok=True)
        save_png(small, os.path.join(OUT, sys.argv[3] if len(sys.argv) > 3 else "delivery/cover_1080x1920.png"), 9)
    elif mode == "still":
        os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
        for s in sys.argv[2:]:
            t = float(s)
            arr, boxes = render(t)
            v = safe_violations(boxes)
            print(f"t={t}: safe-area violations: {v}")
            save_png(arr, os.path.join(OUT, "stills", f"still_{t:06.2f}.png"), 6)
