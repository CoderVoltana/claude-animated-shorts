"""SimTheory #06 renderer: deterministic, offline, frame-by-frame from data/sim.npz + data/physics.json.
Scene (trails, rods, bobs, chart bars) -> float buffers -> bloom pass -> 8-bit -> crisp text/UI layer on top.
Usage:  python3 render.py master            # all 1920 frames -> out/master_video.mkv (lossless FFV1)
        python3 render.py stills 0 300 ...  # PNG stills to stills/
"""
import math, os, re, subprocess, sys
import numpy as np
import cv2
import skia
import timeline as TL

W, H = 1440, 2560
SAFE = (86, 384, 1152, 1664)
FONT_DIR = "/usr/share/fonts/opentype/inter"
TF_BOLD = skia.Typeface.MakeFromFile(f"{FONT_DIR}/Inter-Bold.otf")
TF_REG = skia.Typeface.MakeFromFile(f"{FONT_DIR}/Inter-Regular.otf")
TF_LIGHT = skia.Typeface.MakeFromFile(f"{FONT_DIR}/Inter-Light.otf")

# ---------------------------------------------------------------- layout (all px)
S = 120.0                         # grid scale, px per metre (same for all four)
S_ART = 240.0                     # overlay scale (2x)
BOXES = [(86, 444, 614, 828), (624, 444, 1152, 828), (86, 1280, 614, 1664), (624, 1280, 1152, 1664)]
PIVOTS = [(350.0, 571.0), (888.0, 571.0), (350.0, 1407.0), (888.0, 1407.0)]
ART_PIVOT = (619.0, 844.0)        # 4 m x 2.5 m art box at 240 px/m centred on the safe-area centre
ART_CENTRE = (619.0, 1024.0)
BAND_CY = 1054.0                  # caption strip between the two rows (828..1280)
TOP_CY = 600.0                    # caption slot above the overlay / chart
SIZES = dict(hook=130, caption=96, label=44, sim=44, chart_head=60, chart_label=48)

PHYS, SIM = TL.load()
LABELS = list(PHYS["pendulums"].keys())
TRAJ = SIM["traj"]                # (4, n, 4) th1 th2 w1 w2
FLIPS = SIM["flips"]
LEN = SIM["L"]
HZ = int(SIM["save_hz"])
COL = [tuple(int(PHYS["pendulums"][l]["colour"][i:i + 2], 16) / 255 for i in (1, 3, 5)) for l in LABELS]
CAPS = TL.captions(PHYS)
ROWS = TL.chart_rows(PHYS)
FLIP_EV = TL.flip_events(PHYS)
WHITE = (1.0, 1.0, 1.0)


def hexc(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


# ---------------------------------------------------------------- state helpers
def state(k, s):
    x = s * HZ
    i0 = int(math.floor(x)); f = x - i0
    i0 = min(max(i0, 0), TRAJ.shape[1] - 2)
    return TRAJ[k, i0] * (1 - f) + TRAJ[k, i0 + 1] * f


def bobs_m(k, th1, th2):
    L1, L2 = LEN[k]
    x1, y1 = L1 * math.sin(th1), -L1 * math.cos(th1)
    return (x1, y1), (x1 + L2 * math.sin(th2), y1 - L2 * math.cos(th2))


def path_m(k, s0, s1):
    """Bottom-bob path (metres) between sim times s0 < s1 from the 2400 Hz saved states, with exact interpolated end."""
    i0 = int(math.ceil(s0 * HZ)); i1 = int(math.floor(s1 * HZ))
    th = TRAJ[k, i0:i1 + 1, :2]
    if s1 * HZ - i1 > 1e-9:
        th = np.vstack([th, state(k, s1)[:2]])
    if s0 * HZ < i0 - 1e-9:
        th = np.vstack([state(k, s0)[:2], th])
    L1, L2 = LEN[k]
    x = L1 * np.sin(th[:, 0]) + L2 * np.sin(th[:, 1])
    y = -L1 * np.cos(th[:, 0]) - L2 * np.cos(th[:, 1])
    return np.stack([x, y], 1)


def decimate(px, step=2.0):
    if len(px) < 3:
        return px
    d = np.hypot(*np.diff(px, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] < step:
        return px[[0, -1]]
    targets = np.arange(0, cum[-1], step)
    idx = np.unique(np.concatenate([np.searchsorted(cum, targets), [len(px) - 1]]))
    return px[idx]


def flip_count(k, s):
    return int(FLIPS[k, min(int(math.floor(s * HZ + 1e-9)), FLIPS.shape[1] - 1)])


# ---------------------------------------------------------------- per-time animation parameters
def ramp(t, a, b):
    return TL.smooth((t - a) / (b - a))


def reveal_u(k, t):
    """0 = trail drawing in its cell, 1 = merged into the centred overlay."""
    if t < TL.REWIND:
        return TL.smoother((t - (TL.REVEAL + 0.12 * k)) / 1.5)
    return 1.0 - TL.smoother((t - (TL.REWIND + 0.08 * (3 - k))) / 0.85)


def grid_dim(t):
    if t < TL.REWIND:
        return 1.0 - 0.8 * ramp(t, TL.REVEAL, TL.REVEAL + 1.0)
    return 0.2 + 0.8 * ramp(t, TL.REWIND, TL.REWIND + 0.8)


def art_zoom(t):
    if t < TL.REWIND:
        return 1.0 + 0.06 * TL.smooth((t - TL.REVEAL) / (TL.REWIND - TL.REVEAL))
    return 1.0 + 0.06 * (1.0 - ramp(t, TL.REWIND, TL.REWIND + 0.9))


def art_bright(t):
    if t < TL.REWIND:
        lift = 0.25 * math.sin(math.pi * min(max((t - TL.REVEAL) / 1.9, 0), 1))
        return (1.0 + lift) * (1.0 - 0.78 * ramp(t, TL.CHART, TL.CHART + 0.6))
    return 0.22 + 0.78 * ramp(t, TL.REWIND, TL.REWIND + 0.5)


def chart_alpha(t):
    if t < TL.REWIND:
        return ramp(t, TL.CHART, TL.CHART + 0.3)
    return 1.0 - ramp(t, TL.REWIND, TL.REWIND + 0.45)


def trail_xf(k, t):
    """Pivot (px) and scale (px/m) used to draw pendulum k's trail drawing at time t."""
    u = reveal_u(k, t)
    lift = 1.0 + 0.05 * math.sin(math.pi * u)
    px = PIVOTS[k][0] + (ART_PIVOT[0] - PIVOTS[k][0]) * u
    py = PIVOTS[k][1] + (ART_PIVOT[1] - PIVOTS[k][1]) * u
    sc = (S + (S_ART - S) * u) * lift
    z = art_zoom(t) if u > 0 else 1.0
    z = 1.0 + (z - 1.0) * u
    px = ART_CENTRE[0] + z * (px - ART_CENTRE[0])
    py = ART_CENTRE[1] + z * (py - ART_CENTRE[1])
    return px, py, sc * z


def to_px(m, piv, sc):
    return np.stack([piv[0] + sc * m[:, 0], piv[1] - sc * m[:, 1]], 1)


# ---------------------------------------------------------------- drawing
F16 = skia.ImageInfo.Make(W, H, skia.ColorType.kRGBA_F16_ColorType, skia.AlphaType.kPremul_AlphaType)
TRAIL_SCALE = 8.0                 # skia kPlus clamps at 1.0, so accumulate at 1/8 and scale back
_surf = {}


def surf(name, info=F16):
    if name not in _surf:
        _surf[name] = skia.Surface.MakeRaster(info)
    return _surf[name]


def seg_points(px):
    seg = np.repeat(px, 2, axis=0)[1:-1]
    return [skia.Point(float(a), float(b)) for a, b in seg]


def c4(c, a=1.0, mul=1.0):
    return skia.Color4f(c[0] * mul * a, c[1] * mul * a, c[2] * mul * a, a)


def draw_trails(cv, t, s):
    s_end = min(s, TL.TRAIL_SIM1)
    if s_end <= TL.TRAIL_SIM0 + 1e-6:
        return
    for k in range(4):
        piv_x, piv_y, sc = trail_xf(k, t)
        pts = decimate(to_px(path_m(k, TL.TRAIL_SIM0, s_end), (piv_x, piv_y), sc), 2.0)
        if len(pts) < 2:
            continue
        w = 3.2 * math.sqrt(sc / S)
        a = 0.5 / TRAIL_SCALE
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=w, StrokeCap=skia.Paint.kButt_Cap,
                       BlendMode=skia.BlendMode.kPlus, Color4f=skia.Color4f(COL[k][0] * a, COL[k][1] * a, COL[k][2] * a, a))
        cv.drawPoints(skia.Canvas.kLines_PointMode, seg_points(pts), p)


def glow_dot(cv, x, y, r, col, inten, core_white=0.0):
    halo = skia.Paint(AntiAlias=True)
    halo.setShader(skia.GradientShader.MakeRadial(
        skia.Point(x, y), r * 2.6, [c4(col, 0.55 * inten), c4(col, 0.0)], None))
    cv.drawCircle(x, y, r * 2.6, halo)
    cc = tuple(c * (1 - core_white) + core_white for c in col)
    cv.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Color4f=c4(cc, 1.0, inten)))
    if core_white > 0:
        cv.drawCircle(x, y, r * 0.5, skia.Paint(AntiAlias=True, Color4f=c4(WHITE, 1.0, inten)))


def draw_pendulums(cv, t, s):
    dim = grid_dim(t)
    s_prev = TL.sim_time(max(t - 1 / TL.FPS, 0.0))
    for k in range(4):
        piv = PIVOTS[k]
        th1, th2 = state(k, s)[:2]
        (x1, y1), (x2, y2) = bobs_m(k, th1, th2)
        P1 = (piv[0] + S * x1, piv[1] - S * y1)
        P2 = (piv[0] + S * x2, piv[1] - S * y2)
        # comet tail: last 0.45 s of the path forward; during the rewind the path swept this frame (motion streak)
        if s_prev <= s:
            a0, a1 = max(0.0, s - 0.45), s
        else:
            a0, a1 = s, min(s_prev, s + 0.9)
        if a1 - a0 > 1e-4:
            pts = to_px(path_m(k, a0, a1), piv, S)
            if s_prev > s:
                pts = pts[::-1]          # make the bob end the last point
            pts = decimate(pts, 2.0)
            n = len(pts) - 1
            if n >= 1:
                nb = 10
                for b in range(nb):
                    lo, hi = (n * b) // nb, (n * (b + 1)) // nb
                    if hi <= lo:
                        continue
                    f = (b + 1) / nb
                    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0 + 2.0 * f,
                                   StrokeCap=skia.Paint.kRound_Cap, BlendMode=skia.BlendMode.kPlus,
                                   Color4f=c4(COL[k], 0.5 * f * f * dim))
                    cv.drawPoints(skia.Canvas.kLines_PointMode, seg_points(pts[lo:hi + 1]), p)
        rod = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4.0,
                         StrokeCap=skia.Paint.kRound_Cap, Color4f=c4(tuple(0.55 * c + 0.45 for c in COL[k]), 0.9 * dim))
        cv.drawLine(piv[0], piv[1], P1[0], P1[1], rod)
        cv.drawLine(P1[0], P1[1], P2[0], P2[1], rod)
        cv.drawCircle(piv[0], piv[1], 5.0, skia.Paint(AntiAlias=True, Color4f=c4(WHITE, 0.6 * dim)))
        glow_dot(cv, P1[0], P1[1], 8.0, COL[k], 0.8 * dim)
        glow_dot(cv, P2[0], P2[1], 10.5, COL[k], 1.0 * dim, core_white=0.45)


# chart geometry
CH_X0, CH_X1, CH_ARROW = 230.0, 1030.0, 1094.0
CH_ROWS_Y = [900.0, 1030.0, 1160.0, 1290.0]
CH_BAR_H = 64.0
CH_BASE = 1350.0
CH_HEAD_Y = 790.0


def bar_frac(row, t):
    if t < TL.REWIND:
        return TL.ease_out_cubic((t - row["t0"]) / TL.BAR_GROW)
    return 1.0 - ramp(t, TL.REWIND, TL.REWIND + 0.45)


def bar_len(row):
    return 1.0 if row["tc"] is None else row["tc"] / PHYS["chaos_horizon_s"]


def highlight(j, t):
    """Pulse on the bar named in the current result caption."""
    if 23.5 <= t < 26.0 and j == 0:
        return 0.5 + 0.5 * math.sin(2 * math.pi * 1.5 * (t - 23.5) - math.pi / 2)
    if 26.0 <= t < 29.0 and j == len(ROWS) - 1:
        return 0.5 + 0.5 * math.sin(2 * math.pi * 1.5 * (t - 26.0) - math.pi / 2)
    return 0.0


def draw_chart_bars(cv, t, glow=False):
    """glow=True: faint copy in the scene layer (feeds the bloom); glow=False: crisp exact-colour bars in the UI layer."""
    ca = chart_alpha(t)
    if ca <= 0:
        return
    for j, row in enumerate(ROWS):
        f = bar_frac(row, t)
        if f <= 0:
            continue
        col = hexc(row["colour"])
        y = CH_ROWS_Y[j]
        x_end = CH_X0 + (CH_X1 - CH_X0) * bar_len(row) * f
        hl = highlight(j, t)
        if glow:
            inten, pa = (0.22 + 0.3 * hl) * ca, 1.0
            mk = lambda c, a: c4(c, 1.0, a)
        else:
            col = tuple(c + (1 - c) * 0.18 * hl for c in col)
            inten, pa = 1.0, ca
            mk = lambda c, a: rgba8(c, a)
        rr = skia.RRect.MakeRectXY(skia.Rect(CH_X0, y - CH_BAR_H / 2, max(x_end, CH_X0 + 1), y + CH_BAR_H / 2), 10, 10)
        cv.drawRRect(rr, skia.Paint(AntiAlias=True, Color4f=mk(col, inten * pa) if glow else mk(col, pa)))
        if row["tc"] is None and f > 0.98:
            # drawn arrow: the bar runs off the 60 s edge
            a = TL.smooth((f - 0.98) / 0.02) if t >= TL.REWIND else ramp(t, row["t0"] + TL.BAR_GROW, row["t0"] + TL.BAR_GROW + 0.25)
            nud = 6.0 * hl
            path = skia.Path()
            path.moveTo(CH_X1 + 4 + nud, y - CH_BAR_H / 2 - 16)
            path.lineTo(CH_ARROW + nud, y)
            path.lineTo(CH_X1 + 4 + nud, y + CH_BAR_H / 2 + 16)
            path.close()
            cv.drawPath(path, skia.Paint(AntiAlias=True, Color4f=mk(col, inten * a) if glow else mk(col, pa * a)))


def bloom(img):
    h, w = img.shape[:2]
    acc = np.zeros_like(img)
    cur = img
    for lvl, (sig, wt) in enumerate([(2.5, 0.55), (3.0, 0.45), (3.5, 0.35), (4.0, 0.28), (4.0, 0.2)]):
        cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(cur, (0, 0), sig)
        acc += wt * cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    return img + 0.9 * acc


def soft_clip(x):
    k = 0.75
    y = np.where(x < k, x, k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k))))
    return np.clip(y, 0, 1)


# ---------------------------------------------------------------- text / UI layer (no bloom)
def font(tf, size):
    f = skia.Font(tf, size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    f.setHinting(skia.FontHinting.kNone)
    return f


FONTS = dict(hook=font(TF_BOLD, SIZES["hook"]), cap=font(TF_BOLD, SIZES["caption"]), lab=font(TF_REG, SIZES["label"]),
             labb=font(TF_BOLD, SIZES["label"]), sim=font(TF_LIGHT, SIZES["sim"]),
             head=font(TF_BOLD, SIZES["chart_head"]), rl=font(TF_REG, SIZES["chart_label"]))
CAP_H = {k: f.getMetrics().fCapHeight for k, f in FONTS.items()}


def rgba8(c, a):
    return skia.Color4f(c[0], c[1], c[2], max(0.0, min(1.0, a)))


def text_paint(col, a, shadow=False):
    p = skia.Paint(AntiAlias=True, Color4f=rgba8(col, a))
    if shadow:
        p.setImageFilter(skia.ImageFilters.DropShadow(0, 5, 12, 12, skia.Color4f(0, 0, 0, 0.9 * a)))
    return p


RATIO_RE = re.compile(r"(\d:\d)")


def draw_line_tokens(cv, line, f, cx, baseline, a, shadow, colour_ratio):
    toks = [x for x in RATIO_RE.split(line) if x] if colour_ratio else [line]
    total = sum(f.measureText(x) for x in toks)
    x = cx - total / 2
    for tk in toks:
        col = WHITE
        if colour_ratio and RATIO_RE.fullmatch(tk) and tk in LABELS:
            col = COL[LABELS.index(tk)]
        cv.drawString(tk, x, baseline, f, text_paint(col, a, shadow))
        x += f.measureText(tk)


def draw_block(cv, lines, f, fk, cx, cy, gap, a, scale=1.0, colour_ratio=False):
    capH = CAP_H[fk]
    n = len(lines)
    block = capH + gap * (n - 1)
    cv.save()
    cv.translate(cx, cy); cv.scale(scale, scale); cv.translate(-cx, -cy)
    for i, ln in enumerate(lines):
        base = cy - block / 2 + capH + gap * i
        draw_line_tokens(cv, ln, f, cx, base, a, True, colour_ratio)
    cv.restore()


def caption_alpha(t, t_in, t_out, fade_in):
    if t < t_in or t >= t_out:
        return 0.0
    a = 1.0
    if fade_in:
        a = min(a, (t - t_in) / 0.2)
    a = min(a, (t_out - t) / 0.2)
    return TL.smooth(a)


def hook_alpha_scale(t):
    if t < 2.5:
        return caption_alpha(t, 0.0, 2.5, False), 1.0
    if t >= 31.0:
        u = TL.smoother((t - 31.0) / 0.55)
        return u, 1.04 - 0.04 * u
    return 0.0, 1.0


def pop(k, t):
    """Flip-counter pop: decays 0.3 s after each tick (forward run only)."""
    if t >= TL.REWIND:
        return 0.0
    best = 0.0
    for tv, kk, _ in FLIP_EV:
        if kk == k and 0 <= t - tv < 0.3:
            best = max(best, 1.0 - (t - tv) / 0.3)
    return best


def draw_ui(cv, t, s):
    dim = grid_dim(t)
    ui_a = 0.12 + 0.88 * (dim - 0.2) / 0.8          # labels/borders dim with the grid
    ui_a *= 1.0 - 0.6 * chart_alpha(t)                # and get out of the chart's way
    # cell hairlines
    for (x0, y0, x1, y1) in BOXES:
        cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5), 18, 18),
                     skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0,
                                Color4f=skia.Color4f(1, 1, 1, 0.13 * ui_a)))
    # labels + flip counters
    capL = CAP_H["lab"]
    for k, (x0, y0, x1, y1) in enumerate(BOXES):
        base = y0 + 14 + capL
        cv.drawString(LABELS[k], x0 + 20, base, FONTS["lab"], text_paint(COL[k], ui_a))
        n = str(flip_count(k, s))
        fb = FONTS["labb"]
        wn = fb.measureText(n)
        xr = x1 - 20
        pk = pop(k, t)
        cv.save()
        cx, cy = xr - wn / 2, base - capL / 2
        sc = 1.0 + 0.35 * pk
        cv.translate(cx, cy); cv.scale(sc, sc); cv.translate(-cx, -cy)
        ncol = tuple(c * pk + 1.0 * (1 - pk) for c in COL[k])
        cv.drawString(n, xr - wn, base, fb, text_paint(ncol, ui_a))
        cv.restore()
        wf = FONTS["lab"].measureText("flips")
        cv.drawString("flips", xr - wn - 14 - wf, base, FONTS["lab"], text_paint(WHITE, 0.55 * ui_a))
    # series tag (+ rewind glyph drawn as shapes)
    cv.drawString("SIM #06", 100, 400 + CAP_H["sim"], FONTS["sim"], text_paint(hexc("#22E5FF"), 1.0))
    if TL.REWIND <= t < TL.REWIND_END - 0.15:
        a = 0.55 + 0.45 * math.cos(2 * math.pi * 3 * (t - TL.REWIND))
        x = 100 + FONTS["sim"].measureText("SIM #06") + 26
        yc = 400 + CAP_H["sim"] / 2
        for dx in (0, 26):
            p = skia.Path()
            p.moveTo(x + dx + 26, yc - 15); p.lineTo(x + dx, yc); p.lineTo(x + dx + 26, yc + 15); p.close()
            cv.drawPath(p, skia.Paint(AntiAlias=True, Color4f=rgba8(hexc("#22E5FF"), a)))
    # chart bars (crisp) + text + axis
    ca = chart_alpha(t)
    if ca > 0:
        draw_chart_bars(cv, t)
        axis = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0, Color4f=skia.Color4f(1, 1, 1, 0.35 * ca))
        g = ramp(t, TL.CHART, TL.CHART + 0.4) if t < TL.REWIND else 1.0
        cv.drawLine(CH_X0, CH_BASE, CH_X0, CH_BASE - (CH_BASE - 845) * g, axis)
        cv.drawLine(CH_X0, CH_BASE, CH_X0 + (CH_X1 - CH_X0) * g, CH_BASE, axis)
        for v in (0, 20, 40, 60):
            x = CH_X0 + (CH_X1 - CH_X0) * v / PHYS["chaos_horizon_s"]
            cv.drawLine(x, CH_BASE, x, CH_BASE + 14, axis)
            lab = f"{v} s" if v == 60 else str(v)
            wl = FONTS["lab"].measureText(lab)
            cv.drawString(lab, x - wl / 2, CH_BASE + 30 + CAP_H["lab"], FONTS["lab"], text_paint(WHITE, 0.6 * ca * g))
        hw = FONTS["head"].measureText("Time until unpredictable")
        cv.drawString("Time until unpredictable", 619 - hw / 2, CH_HEAD_Y + CAP_H["head"] / 2, FONTS["head"],
                      text_paint(WHITE, 0.95 * ca, True))
        for j, row in enumerate(ROWS):
            f = bar_frac(row, t)
            col = hexc(row["colour"])
            y = CH_ROWS_Y[j]
            la = ca * (ramp(t, row["t0"] - 0.15, row["t0"] + 0.1) if t < TL.REWIND else 1.0)
            wl = FONTS["rl"].measureText(row["label"])
            cv.drawString(row["label"], CH_X0 - 22 - wl, y + CAP_H["rl"] / 2, FONTS["rl"], text_paint(col, la))
            va = ca * (ramp(t, row["t0"] + 0.55 * TL.BAR_GROW, row["t0"] + TL.BAR_GROW) if t < TL.REWIND else f)
            txt = TL.fmt_chaos(row["tc"])
            if row["tc"] is None:
                wv = FONTS["labb"].measureText(txt)
                cv.drawString(txt, CH_X1 - 22 - wv, y + CAP_H["labb"] / 2, FONTS["labb"], text_paint((0, 0, 0), va))
            else:
                x_end = CH_X0 + (CH_X1 - CH_X0) * bar_len(row) * f
                cv.drawString(txt, x_end + 18, y + CAP_H["lab"] / 2, FONTS["lab"], text_paint(WHITE, va, True))
    # hook + captions
    ha, hs = hook_alpha_scale(t)
    for lines, t_in, t_out, slot, fade_in in CAPS:
        if slot == "hook":
            if ha > 0:
                draw_block(cv, lines, FONTS["hook"], "hook", 619, BAND_CY, 134, ha, hs)
            continue
        a = caption_alpha(t, t_in, t_out, fade_in)
        if a <= 0:
            continue
        cy = BAND_CY if slot == "band" else TOP_CY
        sc = 0.97 + 0.03 * TL.smooth(min((t - t_in) / 0.2, 1.0))
        draw_block(cv, lines, FONTS["cap"], "cap", 619, cy, 112, a, sc, colour_ratio=(slot == "top"))


# ---------------------------------------------------------------- frame
COVER = dict(on=False, dy=236.0)


def render(i):
    t = TL.frame_time(i)
    if t >= TL.REWIND_END:          # loop hold: identical to frame 0 by construction
        t = 0.0
    s = TL.sim_time(t)
    if COVER["on"]:
        return render_cover(t, s)
    tr = surf("trail"); ob = surf("obj")
    tc = tr.getCanvas(); oc = ob.getCanvas()
    tc.clear(skia.Color4f(0, 0, 0, 0)); oc.clear(skia.Color4f(0, 0, 0, 0))
    draw_trails(tc, t, s)
    draw_pendulums(oc, t, s)
    draw_chart_bars(oc, t, glow=True)
    A = tr.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_F16_ColorType)[..., :3].astype(np.float32)
    B = ob.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_F16_ColorType)[..., :3].astype(np.float32)
    trail = (1.0 - np.exp(-1.6 * TRAIL_SCALE * A)) * art_bright(t)
    scene = soft_clip(bloom(trail + B))
    rgba = np.empty((H, W, 4), np.uint8)
    rgba[..., :3] = (scene * 255.0 + 0.5).astype(np.uint8)
    rgba[..., 3] = 255
    ui = surf("ui", skia.ImageInfo.MakeN32Premul(W, H))
    uc = ui.getCanvas()
    uc.clear(skia.Color4f(0, 0, 0, 1))
    uc.drawImage(skia.Image.fromarray(rgba, colorType=skia.ColorType.kRGBA_8888_ColorType), 0, 0)
    draw_ui(uc, t, s)
    out = ui.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
    return np.ascontiguousarray(out[..., :3])


def render_cover(t, s):
    """Cover: the merged overlay art (as at video time t) moved down, with the hook text above it."""
    tr = surf("trail"); tc = tr.getCanvas(); tc.clear(skia.Color4f(0, 0, 0, 0))
    tc.save(); tc.translate(0, COVER["dy"]); draw_trails(tc, t, s); tc.restore()
    A = tr.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_F16_ColorType)[..., :3].astype(np.float32)
    scene = soft_clip(bloom((1.0 - np.exp(-1.6 * TRAIL_SCALE * A)) * 1.15))
    rgba = np.empty((H, W, 4), np.uint8); rgba[..., :3] = (scene * 255 + 0.5).astype(np.uint8); rgba[..., 3] = 255
    ui = surf("ui", skia.ImageInfo.MakeN32Premul(W, H)); uc = ui.getCanvas(); uc.clear(skia.Color4f(0, 0, 0, 1))
    uc.drawImage(skia.Image.fromarray(rgba, colorType=skia.ColorType.kRGBA_8888_ColorType), 0, 0)
    uc.drawString("SIM #06", 100, 400 + CAP_H["sim"], FONTS["sim"], text_paint(hexc("#22E5FF"), 1.0))
    draw_block(uc, CAPS[0][0], FONTS["hook"], "hook", 619, 690, 134, 1.0)
    out = ui.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
    return np.ascontiguousarray(out[..., :3])


def render_bytes(i):
    return render(i).tobytes()


def main():
    mode = sys.argv[1]
    if mode == "stills":
        os.makedirs("stills", exist_ok=True)
        for a in sys.argv[2:]:
            i = int(a)
            cv2.imwrite(f"stills/f{i:04d}.png", cv2.cvtColor(render(i), cv2.COLOR_RGB2BGR))
            print("still", i)
        return
    if mode == "cover":
        COVER["on"] = True
        img = render(int(21.5 * TL.FPS))
        img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)
        cv2.imwrite("out/SimTheory06_cover_1080x1920.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("cover written")
        return
    if mode == "master":
        from multiprocessing import Pool
        os.makedirs("out", exist_ok=True)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(TL.FPS), "-i", "-", "-c:v", "ffv1", "-level", "3",
                               "-pix_fmt", "rgb24", "-slices", "16", "out/master_video.mkv"], stdin=subprocess.PIPE)
        with Pool(4) as pool:
            for n, b in enumerate(pool.imap(render_bytes, range(TL.NFRAMES), chunksize=2)):
                ff.stdin.write(b)
                if n % 120 == 0:
                    print("frame", n, flush=True)
        ff.stdin.close(); ff.wait()
        print("master done")


if __name__ == "__main__":
    main()
