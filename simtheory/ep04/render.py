"""Step 2: deterministic offline frame renderer for SimTheory #04 "Can friction kill chaos?".

Reads ONLY data/sim.npz + data/measure.json (written by simulate.py) and timeline.py (video->sim clock).
Each frame:
  world layer  : each pendulum (rods, bobs, ~1 s trail) is drawn anti-aliased by skia into its own 8-bit
                 coverage mask, then added in float HDR with its colour (additive: aligned twins -> near white)
  bloom        : multi-scale Gaussian bloom of the HDR world (blurred copies added on top)
  tone map     : over-bright channels spill into the others (white-hot cores), soft shoulder
  HUD layer    : screen-space text/bars drawn by skia (premultiplied RGBA) composited on top
Usage: python render.py frames_dir [first last step]   |   python render.py --still t out.png [--cover]
"""
import json, os, sys, math, csv
import numpy as np
import cv2
import skia
import timeline as TL
from physics import energy, E_REST, wrap180

W, H = 1440, 2560
HERE = TL.HERE
D = np.load(os.path.join(HERE, "data", "sim.npz"))
S, DT = D["S"], float(D["dt"])
MEAS = json.load(open(os.path.join(HERE, "data", "measure.json")))
E_AVAIL = MEAS["E_available_J"]

# ------------------------------------------------------------------ layout (px)
SAFE = (86, 384, 1152, 1664)
LPX = 170.0                                # 1 m arm = 170 px  -> full reach 340 px, lane half-width 360 px
PIVOT_Y = 1235.0
PIVOTS = [(360.0, PIVOT_Y), (1080.0, PIVOT_Y)]
COL_X = [360.0, 975.0]                     # HUD column centres (right column + glow kept inside x <= 1152)
CAP_CX, CAP_CY, CAP_MAXW = 619.0, 635.0, 1040.0
HUD_BASE = 432.0
LABEL_BASE, BAR_Y, BAR_H, BAR_W = 850.0, 868.0, 10.0, 300.0
READ_BASE = 1640.0
FOCUS = np.array([1080.0, PIVOT_Y + LPX])  # camera push-in focus (middle of the hanging friction pendulum)
PUSH_TO = np.array([960.0, PIVOT_Y + LPX - 50.0])
ZMAX = 1.38


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


CYAN, BLUE, PURPLE, MAGENTA = hexc("22E5FF"), hexc("3D7BFF"), hexc("9B5CFF"), hexc("FF4FD8")
ORANGE, WHITE, GOLD = hexc("FF8A2A"), hexc("FFFFFF"), hexc("FFC94A")
PCOL = [CYAN, BLUE, ORANGE, MAGENTA]       # pendulums: L-A, L-B, R-A, R-B


def skc(c, a=1.0):
    return skia.Color4f(float(c[0]), float(c[1]), float(c[2]), float(a))


FONT_DIR = "/usr/share/fonts/opentype/inter/"
TF_BOLD = skia.Typeface.MakeFromFile(FONT_DIR + "Inter-Bold.otf")
TF_REG = skia.Typeface.MakeFromFile(FONT_DIR + "Inter-Regular.otf")
TF_LIGHT = skia.Typeface.MakeFromFile(FONT_DIR + "Inter-Light.otf")
assert TF_BOLD and TF_REG and TF_LIGHT


def font(tf, size):
    f = skia.Font(tf, size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    f.setHinting(skia.FontHinting.kNone)
    return f


# ------------------------------------------------------------------ captions (exact on-screen text)
OFFSET_TXT = f"{MEAS['twin_offset_deg']:g}"
CAPTIONS = [
    (2.5, 6.0, f"Each side: two twins, {OFFSET_TXT}° apart."),
    (6.0, 10.0, "Chaos tears the twins apart."),
    (10.5, 14.0, "Friction drains the energy chaos feeds on."),
    (14.0, 17.0, "No energy, no chaos."),
    (17.5, 20.5, "Chaos split them."),
    (20.5, 24.0, "Friction reunited them."),
    (24.0, 28.0, "No friction: chaos never ends."),
    (28.0, 31.0, "So yes. Friction kills chaos."),
]
FADE = 0.2
T_MERGE = MEAS["remerge_right_video"]
T_REST = MEAS["rest_right_video"]


def env_window(t, a, b, fade=FADE, fade_in=True, fade_out=True):
    if t < a or t >= b:
        return 0.0
    v = 1.0
    if fade_in:
        v = min(v, (t - a) / fade)
    if fade_out:
        v = min(v, (b - t) / fade)
    return float(np.clip(v, 0, 1))


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def hook_alpha(t):
    if t < 2.4:
        return 1.0
    if t < 2.6:
        return (2.6 - t) / 0.2
    if t >= 33.4:
        return 1.0
    if t >= 33.0:
        return (t - 33.0) / 0.4
    return 0.0


def camera(t):
    """returns zoom z, screen position P of the world focus point, and push amount e (0..1)"""
    if t < 17.0 or t >= 27.0:
        e = 0.0
    elif t < 22.5:
        e = ease((t - 17.0) / 5.5)
    elif t < 24.0:
        e = 1.0
    else:
        e = 1.0 - ease((t - 24.0) / 3.0)
    z = 1.0 + (ZMAX - 1.0) * e
    P = FOCUS + (PUSH_TO - FOCUS) * e
    return z, P, e


# ------------------------------------------------------------------ sim access
def state_at(s):
    f = min(max(s / DT, 0.0), len(S) - 1.000001)
    i = int(math.floor(f))
    a = f - i
    return S[i] * (1 - a) + S[i + 1] * a


def bob_xy(st, p, pivot):
    th1, th2 = st[p, 0], st[p, 1]
    x1 = pivot[0] + LPX * math.sin(th1)
    y1 = pivot[1] + LPX * math.cos(th1)
    return (x1, y1), (x1 + LPX * math.sin(th2), y1 + LPX * math.cos(th2))


def efrac_at(s):
    st = state_at(s)
    return (energy(st) - E_REST) / E_AVAIL


def trail_window(t, s):
    """returns (s_from, s_to, alpha_mult) sim interval of the trail; newest end = s"""
    if t >= TL.T_RW0:
        u = float(TL.rewind_u(t))
        w = 2.0 * (1.0 - u) ** 1.5
        return s, s + w, (1.0 - u)
    sp = float(TL.sim_time(max(t - 1.0, 0.0)))
    w = min(s - sp, 2.0)
    return s - w, s, 1.0


# ------------------------------------------------------------------ surfaces
class Ctx:
    def __init__(self):
        self.masks = [np.zeros((H, W), np.uint8) for _ in range(5)]  # 4 pendulums + neutral (pivots/divider)
        self.msurf = [skia.Surface(m, colorType=skia.kAlpha_8_ColorType) for m in self.masks]
        self.hud = np.zeros((H, W, 4), np.uint8)
        self.hsurf = skia.Surface(self.hud, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)


def paint_stroke(width, alpha, cap=skia.Paint.kRound_Cap):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=cap)
    p.setColor4f(skia.Color4f(1, 1, 1, alpha))
    return p


def paint_fill(alpha, blur=0.0):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style)
    p.setColor4f(skia.Color4f(1, 1, 1, alpha))
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def draw_pendulum(cv, p, pivot, s, t, z, trail_len_mult, trail_alpha_mult):
    st = state_at(s)
    # --- trail on the bottom bob
    s0, s1, am = trail_window(t, s)
    span = s1 - s0
    if t < TL.T_RW0:
        span *= trail_len_mult
        s0 = s1 - span
    am *= trail_alpha_mult
    if span > 1e-4 and am > 0.003:
        n = int(np.clip(span * 240, 2, 260))
        ts = np.linspace(s0, s1, n + 1)
        f = np.clip(ts / DT, 0, len(S) - 1.000001)
        i = np.floor(f).astype(int)
        a = (f - i)[:, None]
        th = S[i, p, :2] * (1 - a) + S[i + 1, p, :2] * a
        x = pivot[0] + LPX * (np.sin(th[:, 0]) + np.sin(th[:, 1]))
        y = pivot[1] + LPX * (np.cos(th[:, 0]) + np.cos(th[:, 1]))
        newest_is_end = t < TL.T_RW0
        for k in range(n):
            age = (1.0 - (k + 0.5) / n) if newest_is_end else ((k + 0.5) / n)
            al = am * 0.75 * (1.0 - age) ** 1.6
            if al < 0.004:
                continue
            wdt = (3.2 + 1.6 * (1.0 - age))
            cv.drawLine(x[k], y[k], x[k + 1], y[k + 1], paint_stroke(wdt, al))
    # --- rods + bobs
    (x1, y1), (x2, y2) = bob_xy(st, p, pivot)
    path = skia.Path()
    path.moveTo(*pivot)
    path.lineTo(x1, y1)
    path.lineTo(x2, y2)
    rp = paint_stroke(5.0, 0.9)
    rp.setStrokeJoin(skia.Paint.kRound_Join)
    cv.drawPath(path, rp)
    for (bx, by), r in (((x1, y1), 7.5), ((x2, y2), 9.5)):
        cv.drawCircle(bx, by, r * 2.0, paint_fill(0.30, blur=r * 0.9))
        cv.drawCircle(bx, by, r, paint_fill(1.0))


def draw_world(ctx, t, s, z, P, ef_R):
    for m in ctx.masks:
        m[:] = 0
    M = skia.Matrix()
    M.setScaleTranslate(z, z, P[0] - z * FOCUS[0], P[1] - z * FOCUS[1])
    rest_glow = 1.0
    fR = math.sqrt(max(ef_R, 0.0))
    for p in range(4):
        cv = ctx.msurf[p].getCanvas()
        cv.save()
        cv.concat(M)
        lane = p // 2
        if lane == 0:
            draw_pendulum(cv, p, PIVOTS[0], s, t, z, 1.0, 1.0)
        else:
            draw_pendulum(cv, p, PIVOTS[1], s, t, z, 0.25 + 0.75 * fR, 0.3 + 0.7 * fR)
        cv.restore()
    # neutral layer: pivots + hairline divider
    cv = ctx.msurf[4].getCanvas()
    cv.save()
    cv.concat(M)
    dp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0)
    dp.setShader(skia.GradientShader.MakeLinear(
        [skia.Point(720, 880), skia.Point(720, 1600)],
        [skia.ColorSetARGB(0, 255, 255, 255), skia.ColorSetARGB(255, 255, 255, 255), skia.ColorSetARGB(0, 255, 255, 255)],
        [0.0, 0.5, 1.0]))
    cv.drawLine(720, 880, 720, 1600, dp)
    for pv in PIVOTS:
        cv.drawCircle(pv[0], pv[1], 6.0, paint_fill(1.0))
    cv.restore()


def pulse_gain(t):
    """deliberate 'breathing' glow on the resting merged pendulum (so the freeze is never fully still)"""
    a = env_window(t, T_REST + 0.3, TL.T_RW0, fade=0.8)
    return 1.0 + 0.14 * a * math.sin(2 * math.pi * 0.75 * (t - T_REST - 0.3))


def flash_amount(t):
    if t < T_MERGE:
        return 0.0
    dt_ = t - T_MERGE
    return (min(dt_ / 0.08, 1.0)) * math.exp(-max(dt_ - 0.08, 0.0) / 0.55)


def compose_world(ctx, t, z, P):
    hdr = np.zeros((H, W, 3), np.float32)
    pg = pulse_gain(t)
    gains = [0.88, 0.88, 0.88 * pg, 0.88 * pg]
    for p in range(4):
        m = ctx.masks[p]
        ys, xs = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        mm = m[y0:y1, x0:x1].astype(np.float32) * (gains[p] / 255.0)
        hdr[y0:y1, x0:x1] += mm[:, :, None] * PCOL[p][None, None, :]
    hdr += (ctx.masks[4].astype(np.float32) * (0.16 / 255.0))[:, :, None]
    # merge flash: soft gold-white bloom around the fused pendulum
    fl = flash_amount(t)
    if fl > 1e-3:
        c = P + z * (np.array([1080.0, PIVOT_Y + 1.2 * LPX]) - FOCUS)
        r = 130.0 * z
        y0, y1 = int(max(c[1] - 3 * r, 0)), int(min(c[1] + 3 * r, H))
        x0, x1 = int(max(c[0] - 3 * r, 0)), int(min(c[0] + 3 * r, W))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r)) * (0.30 * fl)
        hdr[y0:y1, x0:x1] += g[:, :, None] * (0.55 * GOLD + 0.45 * WHITE)[None, None, :]
    # bloom: blurred copies at 1/2, 1/4, 1/8 resolution added on top
    h2 = cv2.resize(hdr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    h4 = cv2.resize(h2, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    h8 = cv2.resize(h4, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    b2 = cv2.GaussianBlur(h2, (0, 0), 5.0)
    b4 = cv2.GaussianBlur(h4, (0, 0), 6.0)
    b8 = cv2.GaussianBlur(h8, (0, 0), 8.0)
    bloom = cv2.resize(b8, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR) * 0.9 + b4 * 0.8
    bloom = cv2.resize(bloom, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR) + b2 * 0.7
    bloom = cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = 1.0 + 0.35 * fl
    out = hdr + bloom * (0.85 * bg)
    # tone map: spill over-bright energy into all channels (aligned twins -> white-hot), soft shoulder
    mx = out.max(axis=2, keepdims=True)
    out = out + np.maximum(mx - 1.0, 0.0) * 0.5
    k = 0.82
    out = np.where(out > k, k + (1 - k) * (1 - np.exp(-(out - k) / (1 - k))), out)
    return out


# ------------------------------------------------------------------ HUD
def draw_text(cv, txt, x, y, fnt, color, alpha, align="center", shadow=False):
    if alpha <= 0.002:
        return
    w = fnt.measureText(txt)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if shadow:
        sp = skia.Paint(AntiAlias=True)
        sp.setColor4f(skia.Color4f(0, 0, 0, 0.85 * alpha))
        sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, fnt.getSize() * 0.07))
        cv.drawString(txt, x, y + fnt.getSize() * 0.04, fnt, sp)
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(skc(color, alpha))
    cv.drawString(txt, x, y, fnt, p)
    return w


def wrap_lines(txt, fnt, maxw):
    words = txt.split()
    best = None
    for nl in (1, 2, 3):
        # balanced split: minimise the widest line
        import itertools
        n = len(words)
        cand = None
        for cuts in itertools.combinations(range(1, n), nl - 1):
            idx = (0,) + cuts + (n,)
            lines = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(nl)]
            wmax = max(fnt.measureText(l) for l in lines)
            if cand is None or wmax < cand[0]:
                cand = (wmax, lines)
        if cand and cand[0] <= maxw:
            return cand[1]
        best = cand
    return best[1]


F_CAP = font(TF_BOLD, 96)
F_HOOK = font(TF_BOLD, 130)
F_LABEL = font(TF_REG, 44)
F_SIM = font(TF_LIGHT, 44)
F_HUD = font(TF_REG, 42)
F_READ = font(TF_REG, 48)
F_READB = font(TF_BOLD, 48)
_wrap_cache = {}
COVER = False


def caption_block(cv, txt, alpha):
    if txt not in _wrap_cache:
        _wrap_cache[txt] = wrap_lines(txt, F_CAP, CAP_MAXW)
    lines = _wrap_cache[txt]
    lh = 112.0
    cap = F_CAP.getMetrics().fCapHeight
    top = CAP_CY - (lh * (len(lines) - 1) + cap) / 2
    for i, l in enumerate(lines):
        draw_text(cv, l, CAP_CX, top + cap + i * lh, F_CAP, WHITE, alpha, shadow=True)


def hook_block(cv, alpha):
    if alpha <= 0.002:
        return
    lines = [[("Can ", WHITE), ("friction", ORANGE)], [("kill ", WHITE), ("chaos?", CYAN)]]
    lh = 142.0
    cap = F_HOOK.getMetrics().fCapHeight
    top = CAP_CY - (lh + cap) / 2
    for i, segs in enumerate(lines):
        wtot = sum(F_HOOK.measureText(s) for s, _ in segs)
        x = CAP_CX - wtot / 2
        for s_, col in segs:
            draw_text(cv, s_, x, top + cap + i * lh, F_HOOK, col, alpha, align="left", shadow=True)
            x += F_HOOK.measureText(s_)


def rounded_bar(cv, x0, y, w, h, frac, c0, c1, alpha):
    track = skia.Paint(AntiAlias=True)
    track.setColor4f(skia.Color4f(1, 1, 1, 0.14 * alpha))
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x0, y, w, h), h / 2, h / 2), track)
    fw = max(w * float(np.clip(frac, 0, 1)), 0.0)
    if fw < 0.5:
        return
    g = skia.Paint(AntiAlias=True)
    g.setShader(skia.GradientShader.MakeLinear([skia.Point(x0, y), skia.Point(x0 + w, y)],
                                               [skc(c0, alpha).toColor(), skc(c1, alpha).toColor()]))
    glow = skia.Paint(g)
    glow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 7.0))
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x0, y, fw, h), h / 2, h / 2)
    cv.drawRRect(rr, glow)
    cv.drawRRect(rr, g)


def rewind_icon(cv, xr, ybase, size, alpha):
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(skc(GOLD, alpha))
    h = size * 0.72
    for k in range(2):
        x = xr - k * h * 0.85
        path = skia.Path()
        path.moveTo(x, ybase - h)
        path.lineTo(x - h * 0.85, ybase - h / 2)
        path.lineTo(x, ybase)
        path.close()
        cv.drawPath(path, p)
    return 2 * h * 0.85 + 8


def fmt_pct(v):
    return f"{int(round(v))}%"


def draw_hud(ctx, t, s, e_push, efL, efR, log=None):
    ctx.hud[:] = 0
    cv = ctx.hsurf.getCanvas()
    # series tag
    draw_text(cv, "SIM #04", 100, HUD_BASE, F_SIM, CYAN, 1.0, align="left")
    # clock + playback speed (top-right of safe area)
    xr = 1130.0
    if TL.T_RW0 <= t < TL.T_RW1:
        speed_txt, scol = "REWIND", GOLD
        wtxt = draw_text(cv, speed_txt, xr, HUD_BASE, F_HUD, scol, 1.0, align="right")
        xr -= wtxt + 14
        xr -= rewind_icon(cv, xr, HUD_BASE, 42, 1.0)
    else:
        r = float(TL.rate(t)) if t < TL.T_RW0 else 1.0
        r = max(r, 1.0)
        if r < 1.05:
            speed_txt, scol = "1×", WHITE
        else:
            speed_txt, scol = f"TIME-LAPSE {r:.1f}×", GOLD
        wtxt = draw_text(cv, speed_txt, xr, HUD_BASE, F_HUD, scol, 1.0 if r >= 1.05 else 0.75, align="right")
        xr -= wtxt
    clock = f"t = {s:.1f} s"
    draw_text(cv, clock, xr - 30, HUD_BASE, F_HUD, WHITE, 0.6, align="right")
    # lane labels + energy bars
    la = 1.0 - 0.75 * e_push
    draw_text(cv, "NO FRICTION", COL_X[0], LABEL_BASE, F_LABEL, WHITE, 0.92 * la)
    draw_text(cv, "FRICTION", COL_X[1], LABEL_BASE, F_LABEL, WHITE, 0.92)
    rounded_bar(cv, COL_X[0] - BAR_W / 2, BAR_Y, BAR_W, BAR_H, efL, CYAN, BLUE, la)
    rounded_bar(cv, COL_X[1] - BAR_W / 2, BAR_Y, BAR_W, BAR_H, efR, ORANGE, MAGENTA, 1.0)
    # hook / captions
    hook_block(cv, 1.0 if COVER else hook_alpha(t))
    cap_txt = ""
    for a, b, txt in ([] if COVER else CAPTIONS):
        al = env_window(t, a, b)
        if al > 0:
            caption_block(cv, txt, al)
            cap_txt = txt
    # energy readouts (28-31 s)
    ra = 0.0 if COVER else env_window(t, 28.0, 31.0)
    rd = ["", ""]
    if ra > 0:
        for lane, (cx, ef, col) in enumerate(((COL_X[0], efL, CYAN), (COL_X[1] - 50, efR, ORANGE))):
            a_txt, b_txt = "Energy left: ", fmt_pct(100 * ef)
            wa, wb = F_READ.measureText(a_txt), F_READB.measureText(b_txt)
            x = cx - (wa + wb) / 2
            draw_text(cv, a_txt, x, READ_BASE, F_READ, WHITE, 0.9 * ra, align="left", shadow=True)
            draw_text(cv, b_txt, x + wa, READ_BASE, F_READB, GOLD, ra, align="left", shadow=True)
            rd[lane] = a_txt + b_txt
    if log is not None:
        log.update(dict(clock=clock.strip(), speed=speed_txt, caption=cap_txt, hook=round(hook_alpha(t), 3),
                        readout_left=rd[0], readout_right=rd[1], bar_left=round(float(efL), 6), bar_right=round(float(efR), 6)))


def render_frame(ctx, t, log=None):
    s = float(TL.sim_time(t))
    z, P, e = camera(t)
    ef = efrac_at(s)
    efL, efR = float(ef[:2].mean()), float(ef[2:].mean())
    draw_world(ctx, t, s, z, P, efR)
    world = compose_world(ctx, t, z, P)
    draw_hud(ctx, t, s, e, efL, efR, log)
    hud = ctx.hud.astype(np.float32) / 255.0
    rgb = hud[:, :, :3] + np.clip(world, 0, 1) * (1.0 - hud[:, :, 3:4])
    img = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if log is not None:
        log.update(dict(sim_t=round(s, 5), zoom=round(z, 5)))
    return img


def run_range(args):
    outdir, frames = args
    ctx = Ctx()
    rows = []
    for k in frames:
        t = k / TL.FPS
        lg = {"frame": k, "video_t": round(t, 5)}
        img = render_frame(ctx, t, lg)
        cv2.imwrite(os.path.join(outdir, f"{k:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        rows.append(lg)
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--still":
        t = float(sys.argv[2])
        ctx = Ctx()
        COVER = "--cover" in sys.argv
        img = render_frame(ctx, t)
        cv2.imwrite(sys.argv[3], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sys.exit(0)
    from multiprocessing import Pool
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    last = int(sys.argv[3]) if len(sys.argv) > 3 else TL.NFR - 1
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    frames = list(range(first, last + 1, step))
    nproc = int(os.environ.get("NPROC", "4"))
    chunks = [(outdir, frames[i::nproc]) for i in range(nproc)]
    with Pool(nproc) as pool:
        res = pool.map(run_range, chunks)
    rows = sorted([r for rr in res for r in rr], key=lambda r: r["frame"])
    with open(os.path.join(HERE, "data", "hud_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("rendered", len(rows))
