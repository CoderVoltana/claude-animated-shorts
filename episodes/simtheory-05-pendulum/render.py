"""Step 3: deterministic offline renderer (Skia, anti-aliased vector drawing + multi-scale bloom pass).

Every frame is a pure function of its frame index and the saved simulation data (data/sim.npz).
Usage:
    python render.py stills 0 30 600 ...         -> stills/fXXXX.png
    python render.py video out/main_noaudio.mp4    -> raw RGBA frames piped into ffmpeg (no audio)
"""
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import skia

import timeline as TL
from config import (BLUE, COLORS, CYAN, DT, FONT_BOLD, FONT_LIGHT, FONT_MONO, FONT_REG, FPS, FRACS, GOLD, H, HOOK,
                    MAGENTA, NFRAMES, NUM, ORANGE, PIVOT, PURPLE, R, SAFE, T_SIM, THR, TOP, W, W0, WHITE)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"


# ====================================================================== data access
class Data:
    def __init__(self):
        z = np.load(os.path.join(HERE, "data", "sim.npz"))
        self.th = z["theta"]
        self.om = z["omega"]
        self.n = self.th.shape[0] - 1
        ev = json.load(open(os.path.join(HERE, "data", "events_sim.json")))
        self.tops = sorted((t, int(i)) for i, ts in ev["top"].items() for t in ts)
        # phase-curve samples at 240 Hz from the map start (sim 17.5 s) to 27.5 s
        k0 = int(round(TL.CURVE_A / DT))
        idx = k0 + 20 * np.arange(0, 2401)
        self.curve_t = TL.CURVE_A + np.arange(0, 2401) / 240.0
        self.curve_th = self.th[idx]
        self.curve_om = self.om[idx]

    def state(self, i, t):
        """theta, omega of pendulum i at sim time(s) t. Numeric pendulums: cubic Hermite on the RK4 states
        (exact on the 4800 Hz grid); threshold pendulum: exact analytic formula."""
        t = np.asarray(t, float)
        if i == THR:
            d = 4.0 * np.arctan(np.exp(-W0 * t))
            return math.pi - d, 2.0 * W0 / np.cosh(W0 * t)
        t = np.clip(t, 0.0, T_SIM)
        f = t / DT
        j = np.minimum(np.floor(f).astype(int), self.n - 1)
        u = f - j
        p0, p1 = self.th[j, i], self.th[j + 1, i]
        w0, w1 = self.om[j, i], self.om[j + 1, i]
        m0, m1 = w0 * DT, w1 * DT
        u2, u3 = u * u, u * u * u
        th = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
        om = w0 + (w1 - w0) * u
        return th, om


# ====================================================================== drawing helpers
def c4(rgb, a=1.0):
    return skia.Color4f(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(max(0.0, min(1.0, a))))


def stroke(rgb, a, w, cap=skia.Paint.kRound_Cap):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(w), StrokeCap=cap,
                   StrokeJoin=skia.Paint.kRound_Join)
    p.setColor4f(c4(rgb, a))
    return p


def fill(rgb, a):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style)
    p.setColor4f(c4(rgb, a))
    return p


def lw(base, k):
    """Line width scaled with the camera, never thinner than 3 px."""
    return max(3.0, base * math.sqrt(k))


def to_screen(cam, p):
    k, F, C = cam
    return C + k * (np.asarray(p) - F)


def glow_dot(cv, x, y, rgb, a, r_core, r_halo):
    if a <= 0.003:
        return
    col_in = c4(rgb, 0.55 * a).toColor()
    col_out = c4(rgb, 0.0).toColor()
    sh = skia.GradientShader.MakeRadial(skia.Point(x, y), r_halo, [col_in, col_out])
    p = skia.Paint(AntiAlias=True, Shader=sh)
    cv.drawCircle(x, y, r_halo, p)
    cv.drawCircle(x, y, r_core, fill(rgb, a))
    cv.drawCircle(x, y, r_core * 0.5, fill(rgb * 0.35 + 0.65, a))


class Fonts:
    def __init__(self):
        tf = skia.Typeface.MakeFromFile
        self.bold, self.reg, self.light, self.mono, self.semi = (tf(FONT_BOLD), tf(FONT_REG), tf(FONT_LIGHT),
                                                                 tf(FONT_MONO), tf(FONT_SEMI))

    def f(self, which, size):
        ft = skia.Font(getattr(self, which), size)
        ft.setEdging(skia.Font.Edging.kAntiAlias)
        ft.setSubpixel(True)
        return ft


TEXTLOG = []


def draw_text(cv, text, font, x, y, rgb=WHITE, a=1.0, align="left", shadow=1.0):
    """y = baseline. Soft dark shadow underneath for readability over glowing lines."""
    if a <= 0.003:
        return 0
    bnd = skia.Rect()
    w = font.measureText(text, skia.TextEncoding.kUTF8, bnd)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if shadow > 0:
        sp = skia.Paint(AntiAlias=True, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14))
        sp.setColor4f(skia.Color4f(0, 0, 0, 0.85 * a * shadow))
        cv.drawString(text, x, y + 5, font, sp)
        sp2 = skia.Paint(AntiAlias=True, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 5))
        sp2.setColor4f(skia.Color4f(0, 0, 0, 0.7 * a * shadow))
        cv.drawString(text, x, y + 3, font, sp2)
    cv.drawString(text, x, y, font, fill(rgb, a))
    if text.strip():
        TEXTLOG.append(dict(text=text, size=round(font.getSize(), 1), alpha=round(a, 3),
                            box=[round(x + bnd.left(), 1), round(y + bnd.top(), 1), round(x + bnd.right(), 1),
                                 round(y + bnd.bottom(), 1)]))
    return w


# ====================================================================== the renderer
class Renderer:
    def __init__(self):
        self.d = Data()
        self.F = Fonts()
        self.info = skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        self.surf = skia.Surface.MakeRaster(self.info)
        self.half = skia.Surface(W // 2, H // 2)
        self.q = skia.Surface(W // 4, H // 4)
        self.qb = skia.Surface(W // 4, H // 4)
        self.e = skia.Surface(W // 8, H // 8)
        self.eb = skia.Surface(W // 8, H // 8)
        self.lin = skia.SamplingOptions(skia.FilterMode.kLinear)
        self.log = {}
        self.cover = False

    # ------------------------------------------------------------ pendulum world
    def bob_world(self, th):
        th = np.asarray(th)
        return np.stack([PIVOT[0] + R * np.sin(th), PIVOT[1] + R * np.cos(th)], axis=-1)

    def draw_stage(self, cv, tv, cam):
        k = cam[0]
        # faint dashed circle (full bob path)
        c = to_screen(cam, PIVOT)
        p = stroke(WHITE, 0.30, lw(3.0, k))
        p.setPathEffect(skia.DashPathEffect.Make([22 * k, 17 * k], 0))
        cv.drawCircle(float(c[0]), float(c[1]), R * k, p)
        # top marker: small gold chevron + tiny ring at the very top
        tp = to_screen(cam, TOP)
        s = max(0.45, k ** 0.6)
        path = skia.Path()
        path.moveTo(tp[0] - 15 * s, tp[1] - 44 * s)
        path.lineTo(tp[0] + 15 * s, tp[1] - 44 * s)
        path.lineTo(tp[0], tp[1] - 20 * s)
        path.close()
        cv.drawPath(path, fill(GOLD, 0.95))
        cv.drawCircle(float(tp[0]), float(tp[1]), 10 * s, stroke(GOLD, 0.75, lw(3.0, k * 0.7)))
        # flashes whenever a flipper passes over the top (12.5-17.5 s)
        fw = TL.flash_window(tv)
        if fw > 0:
            ts = tv - TL.T_PUSH
            for te, i in self.d.tops:
                age = ts - te
                if 0 <= age < 0.5:
                    q = age / 0.5
                    cv.drawCircle(float(tp[0]), float(tp[1]), (14 + 70 * q) * s,
                                  stroke(COLORS[i], fw * (1 - q) ** 1.5 * 0.9, lw(5 * (1 - q) + 3, k)))
        # turning-point ticks for the trapped swingers (8-12.5 s)
        ta = TL.trapped_ticks_alpha(tv)
        if ta > 0:
            for i, f in enumerate(FRACS):
                if f >= 1:
                    continue
                thm = math.acos(1 - 2 * f)
                for sg in (-1, 1):
                    a = sg * thm
                    p0 = to_screen(cam, PIVOT + (R + 22) * np.array([math.sin(a), math.cos(a)]))
                    p1 = to_screen(cam, PIVOT + (R + 52) * np.array([math.sin(a), math.cos(a)]))
                    cv.drawLine(*map(float, p0), *map(float, p1), stroke(COLORS[i], ta, lw(6, k)))

    def draw_pendulum(self, cv, i, t, a, rate, cam, mode, lay):
        k = cam[0]
        rgb = COLORS[i]
        thick = 1.45 if i == THR else 1.0
        pv = to_screen(cam, PIVOT)
        # ---- trail
        if mode in ("live", "replay") and t > 0 and rate > 0:
            span = min(0.7 * rate, t)
            n = 44
            ts = np.linspace(t - span, t, n + 1)
            th, _ = self.d.state(i, ts)
            pts = to_screen(cam, self.bob_world(th))
            for j in range(n):
                q = (j + 1) / n
                al = a * 0.8 * q ** 1.7
                if al < 0.01:
                    continue
                cv.drawLine(float(pts[j, 0]), float(pts[j, 1]), float(pts[j + 1, 0]), float(pts[j + 1, 1]),
                            stroke(rgb, al, lw((2.2 + 3.0 * q) * thick, k)))
        if mode == "rewind":
            u = lay["u"]
            fade = (1 - u) ** 1.5
            # trail = the most recently *displayed* motion, i.e. later sim times; it shortens and fades
            span = 0.7 * (1 - u)
            if span > 1e-3 and fade > 0.01:
                n = 36
                ts = np.linspace(t + span, t, n + 1)
                th, _ = self.d.state(i, np.minimum(ts, TL.LIVE_END))
                pts = to_screen(cam, self.bob_world(th))
                for j in range(n):
                    q = (j + 1) / n
                    cv.drawLine(float(pts[j, 0]), float(pts[j, 1]), float(pts[j + 1, 0]), float(pts[j + 1, 1]),
                                stroke(rgb, a * 0.7 * fade * q ** 1.7, lw((2.2 + 3.0 * q) * thick, k)))
            # motion blur of the rod over this frame's sim interval
            nsub = 7
            ts = np.linspace(lay["t_prev"], t, nsub + 1)[:-1]
            th, _ = self.d.state(i, ts)
            bw = to_screen(cam, self.bob_world(th))
            for j in range(nsub):
                cv.drawLine(float(pv[0]), float(pv[1]), float(bw[j, 0]), float(bw[j, 1]),
                            stroke(rgb, a * 0.16, lw(4.0 * thick, k)))
        # ---- rod + bob
        th, _ = self.d.state(i, t)
        b = to_screen(cam, self.bob_world(th))
        cv.drawLine(float(pv[0]), float(pv[1]), float(b[0]), float(b[1]), stroke(rgb, a * 0.92, lw(4.0 * thick, k)))
        glow_dot(cv, float(b[0]), float(b[1]), rgb, a, max(6.0, 12.0 * thick * math.sqrt(k)),
                 max(12.0, 30.0 * thick * math.sqrt(k)))

    def draw_pendulums(self, cv, tv, cam):
        lays = TL.layers(tv)
        order = NUM + [THR]
        for lay in lays:
            for i in order:
                if i == THR and not lay["thr"]:
                    continue
                w = lay["thr_w"] if i == THR else lay["w"]
                a = w * TL.pend_alpha(i, tv)
                if a <= 0.003:
                    continue
                self.draw_pendulum(cv, i, lay["t"], a, lay["rate"], cam, lay["mode"], lay)
        # pivot
        pv = to_screen(cam, PIVOT)
        glow_dot(cv, float(pv[0]), float(pv[1]), WHITE, 0.9, max(5.0, 8 * math.sqrt(cam[0])), max(10.0, 18 * math.sqrt(cam[0])))

    # ------------------------------------------------------------ phase portrait
    @staticmethod
    def pmap(rect, thdeg, om):
        x0, y0, x1, y1 = rect
        return (x0 + (np.asarray(thdeg) + 180.0) / 360.0 * (x1 - x0),
                y0 + (TL.OM_TOP - np.asarray(om)) / (TL.OM_TOP - TL.OM_BOT) * (y1 - y0))

    def curve_path(self, rect, th, om):
        thw = np.degrees((th + math.pi) % (2 * math.pi) - math.pi)
        xs, ys = self.pmap(rect, thw, om)
        path = skia.Path()
        brk = np.abs(np.diff(thw)) > 180
        path.moveTo(float(xs[0]), float(ys[0]))
        for j in range(1, len(xs)):
            if brk[j - 1]:
                path.moveTo(float(xs[j]), float(ys[j]))
            else:
                path.lineTo(float(xs[j]), float(ys[j]))
        return path

    def draw_plot(self, cv, tv):
        pa = TL.plot_alpha(tv)
        if pa <= 0.003:
            return None
        rect, ms = TL.plot_rect(tv)
        x0, y0, x1, y1 = rect
        scale = (x1 - x0) / (TL.FULL_RECT[2] - TL.FULL_RECT[0])
        # backing panel for the mini version
        if ms > 0:
            pad = 22
            rr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x0 - pad, y0 - pad, x1 + pad, y1 + pad), 22, 22)
            cv.drawRRect(rr, fill(np.zeros(3), 0.82 * ms * pa))
            cv.drawRRect(rr, stroke(WHITE, 0.22 * ms * pa, 3))
        # axes (hairline) with arrowheads
        ax_a = 0.5 * pa
        zx, zy = self.pmap(rect, 0.0, 0.0)
        ext = 20 * scale + 4
        cv.drawLine(float(x0 - ext), float(zy), float(x1 + ext), float(zy), stroke(WHITE, ax_a, 3, skia.Paint.kButt_Cap))
        ytop = y0 - 6 * scale
        cv.drawLine(float(zx), float(y1), float(zx), float(ytop), stroke(WHITE, ax_a, 3, skia.Paint.kButt_Cap))
        ah = max(8.0, 16 * scale)
        tri = skia.Path()
        tri.moveTo(x1 + ext + ah, zy); tri.lineTo(x1 + ext, zy - ah * 0.55); tri.lineTo(x1 + ext, zy + ah * 0.55); tri.close()
        tri.moveTo(zx, ytop - ah); tri.lineTo(zx - ah * 0.55, ytop); tri.lineTo(zx + ah * 0.55, ytop); tri.close()
        cv.drawPath(tri, fill(WHITE, ax_a))
        # ticks at +-180 deg (the top)
        for th in (-180.0, 180.0):
            tx, _ = self.pmap(rect, th, 0.0)
            cv.drawLine(float(tx), float(zy - 12 * scale - 2), float(tx), float(zy + 12 * scale + 2), stroke(WHITE, ax_a, 3))

        live = tv - TL.T_PUSH
        dim = TL.curves_dim(tv) * (1 - 0.45 * ms)
        nmax = len(self.d.curve_t)
        n_now = int(np.clip(np.floor((min(live, 27.5) - TL.CURVE_A) * 240.0), 0, nmax - 1)) + 1
        cw = lw(4.5, scale) if ms < 1 else 3.0
        for i in NUM:
            n = nmax if ms > 0 else n_now
            if n < 2:
                continue
            path = self.curve_path(rect, self.d.curve_th[:n, i], self.d.curve_om[:n, i])
            cv.drawPath(path, stroke(COLORS[i], 0.95 * pa * dim, cw))
        # live dots of the 8 simulated pendulums (full map only)
        da = pa * (1 - ms)
        if da > 0.003 and live <= 27.6:
            for i in NUM:
                th, om = self.d.state(i, live)
                thw = math.degrees((th + math.pi) % (2 * math.pi) - math.pi)
                x, y = self.pmap(rect, thw, om)
                glow_dot(cv, float(x), float(y), COLORS[i], da, 9.0, 26.0)
        # separatrix: omega = +-2 w0 cos(theta/2), exact formula
        prog = 1.0 if ms > 0 else TL.separatrix_progress(tv)
        if prog > 0:
            n = max(2, int(361 * prog))
            thd = np.linspace(-180.0, -180.0 + 360.0 * prog, n)
            wsep = 2 * W0 * np.cos(np.radians(thd) / 2)
            glow = TL.separatrix_glow(tv) if ms == 0 else 0.6
            for sgn in (1, -1):
                xs, ys = self.pmap(rect, thd, sgn * wsep)
                path = skia.Path()
                path.moveTo(float(xs[0]), float(ys[0]))
                for j in range(1, n):
                    path.lineTo(float(xs[j]), float(ys[j]))
                if glow > 0:
                    cv.drawPath(path, stroke(WHITE, pa * 0.22 * min(glow, 1.3), (22 if ms == 0 else 12) * max(scale, 0.5)))
                cv.drawPath(path, stroke(WHITE, pa, (5.0 + 2.5 * min(glow, 1.0)) if ms == 0 else 4.0))
                if prog < 1:
                    glow_dot(cv, float(xs[-1]), float(ys[-1]), WHITE, pa, 8.0, 34.0)
        # threshold pendulum's phase point: live (sitting at the top corner) -> replay (sliding along the boundary)
        if ms < 1 and live <= 28.3:
            th, om = self.d.state(THR, live)
            thw = 180.0 - math.degrees(4.0 * math.atan(math.exp(-W0 * live)))   # gap computed directly
            x, y = self.pmap(rect, thw, om)
            glow_dot(cv, float(x), float(y), WHITE, pa * (1 - ms), 10.0, 30.0)
        if ms > 0:
            lab = TL.readout_clock(tv)
            tau = lab[1] if lab else (TL.LIVE_END if tv >= TL.REW_A else 0.0)
            if lab and lab[0] == "replay":
                tr = np.linspace(max(0.0, tau - 0.35), tau, 24)
                th, om = self.d.state(THR, tr)
                xs, ys = self.pmap(rect, np.degrees(th), om)
                for j in range(len(tr) - 1):
                    q = (j + 1) / len(tr)
                    cv.drawLine(float(xs[j]), float(ys[j]), float(xs[j + 1]), float(ys[j + 1]), stroke(WHITE, pa * ms * q, 4))
            th, om = self.d.state(THR, tau)
            x, y = self.pmap(rect, np.degrees(th), om)
            glow_dot(cv, float(x), float(y), WHITE, pa * ms, 9.0, 30.0)
        return rect, ms, pa

    # ------------------------------------------------------------ legend (glow part)
    def draw_legend_bar(self, cv, tv, geo):
        a = TL.legend_alpha(tv)
        if a <= 0.003:
            return
        bx0, bx1, by = geo
        cols = [c4(c, 0.75 * a).toColor() for c in (BLUE, CYAN, PURPLE, MAGENTA, ORANGE)]
        sh = skia.GradientShader.MakeLinear([skia.Point(bx0, by), skia.Point(bx1, by)], cols)
        p = skia.Paint(AntiAlias=True, Shader=sh, Style=skia.Paint.kStroke_Style, StrokeWidth=6, StrokeCap=skia.Paint.kRound_Cap)
        cv.drawLine(bx0, by, bx1, by, p)
        for i, f in enumerate(FRACS):
            x = bx0 + (f - 0.2) / 1.5 * (bx1 - bx0)
            if i == THR:
                cv.drawCircle(x, by, 13, stroke(WHITE, a, 4))
            glow_dot(cv, x, by, COLORS[i], a, 8.0 if i != THR else 9.0, 20.0)

    # ------------------------------------------------------------ bloom
    def bloom(self):
        img = self.surf.makeImageSnapshot()
        hc = self.half.getCanvas()
        hc.clear(skia.ColorBLACK)
        hc.drawImageRect(img, skia.Rect.MakeWH(W // 2, H // 2), self.lin)
        himg = self.half.makeImageSnapshot()
        qc = self.q.getCanvas()
        qc.clear(skia.ColorBLACK)
        qc.drawImageRect(himg, skia.Rect.MakeWH(W // 4, H // 4), self.lin)
        qimg = self.q.makeImageSnapshot()
        ec = self.e.getCanvas()
        ec.clear(skia.ColorBLACK)
        ec.drawImageRect(qimg, skia.Rect.MakeWH(W // 8, H // 8), self.lin)
        eimg = self.e.makeImageSnapshot()
        qbc = self.qb.getCanvas()
        qbc.clear(skia.ColorBLACK)
        qbc.drawImage(qimg, 0, 0, self.lin, skia.Paint(ImageFilter=skia.ImageFilters.Blur(3.2, 3.2, skia.TileMode.kDecal)))
        ebc = self.eb.getCanvas()
        ebc.clear(skia.ColorBLACK)
        ebc.drawImage(eimg, 0, 0, self.lin, skia.Paint(ImageFilter=skia.ImageFilters.Blur(4.5, 4.5, skia.TileMode.kDecal)))
        cv = self.surf.getCanvas()
        full = skia.Rect.MakeWH(W, H)
        cv.drawImageRect(self.qb.makeImageSnapshot(), full, self.lin, skia.Paint(BlendMode=skia.BlendMode.kPlus, Alphaf=0.95))
        cv.drawImageRect(self.eb.makeImageSnapshot(), full, self.lin, skia.Paint(BlendMode=skia.BlendMode.kPlus, Alphaf=0.85))

    # ------------------------------------------------------------ text layer
    def legend_geo(self):
        f = self.F.f("reg", 44)
        lw_ = f.measureText("LOW ENERGY")
        hw = f.measureText("HIGH")
        x0, x1 = SAFE["x0"] + 14, SAFE["x1"] - 12
        return x0 + lw_ + 34, x1 - hw - 34, 1636.0

    MAXW = 1020.0

    def caption_font(self, txt, size=96.0, which="bold"):
        """Caption font at ~96 px, shrunk only if a line would leave the safe area."""
        f = self.F.f(which, size)
        mw = max(f.measureText(l) for l in self.wrap_lines(txt))
        if mw > self.MAXW:
            f = self.F.f(which, size * self.MAXW / mw)
        return f

    def wrap_lines(self, text):
        return text.split("\n")

    def draw_texts(self, cv, tv, plotinfo):
        F = self.F
        # series branding
        draw_text(cv, "SIM #05", F.f("light", 44), 100, 432, CYAN, 1.0, shadow=0.6)
        # hook
        ha = 1.0 if self.cover else TL.hook_alpha(tv)
        if ha > 0:
            fh = self.caption_font("Can a pendulum\nflip forever?", 112.0 if self.cover else 130.0)
            y0, dy = (632, 124) if self.cover else (680, 150)
            for j, line in enumerate(["Can a pendulum", "flip forever?"]):
                draw_text(cv, line, fh, PIVOT[0], y0 + j * dy, WHITE, ha, "center")
        # captions
        top = 640 if (TL.REPLAY_A - 0.5 <= tv) else 630
        for txt, a, t0 in ([] if self.cover else TL.caption_alphas(tv)):
            fc = self.caption_font(txt)
            rise = 14 * (1 - TL.smooth((tv - t0) / 0.25))
            for j, line in enumerate(self.wrap_lines(txt)):
                draw_text(cv, line, fc, PIVOT[0], top + rise + j * 112, WHITE, a, "center")
        # legend labels
        la = TL.legend_alpha(tv)
        if la > 0:
            f = F.f("reg", 44)
            draw_text(cv, "LOW ENERGY", f, SAFE["x0"] + 14, 1652, WHITE, 0.85 * la, shadow=0.5)
            draw_text(cv, "HIGH", f, SAFE["x1"] - 12, 1652, WHITE, 0.85 * la, "right", shadow=0.5)
        # phase-portrait labels
        if plotinfo:
            rect, ms, pa = plotinfo
            a = pa * (1 - ms)
            if a > 0:
                f = F.f("reg", 44)
                zx, zy = self.pmap(rect, 0.0, 0.0)
                draw_text(cv, "angle", f, rect[2] + 34, zy + 15, WHITE, 0.9 * a, shadow=0.6)
                draw_text(cv, "spin speed", f, zx + 22, rect[1] + 30, WHITE, 0.9 * a, shadow=0.6)
                ea = TL.eye_labels_alpha(tv) * a
                if ea > 0:
                    x, y = self.pmap(rect, 0.0, -1.45)
                    draw_text(cv, "trapped", f, x, y + 16, CYAN, ea, "center")
                    x, y = self.pmap(rect, -170.0, 8.6)
                    draw_text(cv, "flipping", f, x, y + 16, ORANGE, ea)
        # replay / live labels + clock
        ra, va = TL.replay_label_alpha(tv), TL.live_label_alpha(tv)
        fl = F.f("semi", 44)
        fm = F.f("mono", 48)
        row = 506
        lab = TL.readout_clock(tv)
        if ra > 0:
            cv.drawCircle(112, row - 16, 11, fill(GOLD, ra))
            draw_text(cv, "REPLAY · 0.5× SLOW MOTION", fl, 136, row, GOLD, ra, shadow=0.6)
            tau = (tv - TL.REPLAY_A) * TL.REPLAY_RATE if tv >= TL.REPLAY_A else 0.0
            tau = min(max(tau, 0.0), 5.0)
            draw_text(cv, f"t = {tau:.2f} s", fm, SAFE["x1"] - 12, row, WHITE, ra, "right", shadow=0.6)
        if va > 0:
            cv.drawCircle(112, row - 16, 11, fill(MAGENTA, va))
            draw_text(cv, "LIVE · SINCE THE PUSH", fl, 136, row, MAGENTA, va, shadow=0.6)
            tl = min(tv, TL.REW_A) - TL.T_PUSH
            draw_text(cv, f"t = {tl:.2f} s", fm, SAFE["x1"] - 12, row, WHITE, va, "right", shadow=0.6)
        # gap readout (computed directly from delta)
        oa = TL.readout_alpha(tv)
        if oa > 0:
            tclk = lab[1] if lab else (0.0 if tv < 30 else TL.LIVE_END)
            m, e = TL.gap_parts(tclk)
            x = 330
            draw_text(cv, "Gap to top:", F.f("reg", 50), x, 850, WHITE, 0.85 * oa)
            fv = F.f("mono", 80)
            if e is None:
                draw_text(cv, m + "°", fv, x, 942, WHITE, oa)
            else:
                w1 = draw_text(cv, m, fv, x, 942, WHITE, oa)
                fe = F.f("mono", 52)
                w2 = draw_text(cv, e, fe, x + w1 + 4, 904, WHITE, oa)
                w3 = draw_text(cv, " °", fv, x + w1 + 4 + w2, 942, WHITE, oa)
                self.log["readout_right_px"] = x + w1 + 4 + w2 + w3
            self.log["readout"] = TL.gap_string(tclk)
            self.log["clock"] = tclk

    # ------------------------------------------------------------ frame
    def frame(self, fi):
        tv = TL.frame_time(fi)
        self.log = {"frame": fi, "tv": tv}
        TEXTLOG.clear()
        cv = self.surf.getCanvas()
        cv.clear(skia.ColorBLACK)
        cam = TL.camera(tv)
        self.draw_stage(cv, tv, cam)
        self.draw_pendulums(cv, tv, cam)
        geo = self.legend_geo()
        self.draw_legend_bar(cv, tv, geo)
        plotinfo = self.draw_plot(cv, tv)
        self.bloom()
        self.draw_texts(cv, tv, plotinfo)
        self.log["text"] = list(TEXTLOG)
        return self.surf.toarray(colorType=skia.kRGBA_8888_ColorType)


_R = None


def _init():
    global _R
    _R = Renderer()


def _work(fi):
    arr = _R.frame(fi)
    return fi, arr.tobytes(), dict(_R.log)


def render_video(out_path, frames=None, workers=4):
    frames = list(range(NFRAMES)) if frames is None else frames
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "ffv1", "-level", "3", "-threads", "4", "-slices", "16", "-pix_fmt", "bgr0",
           out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    logs = []
    with Pool(workers, initializer=_init) as pool:
        for fi, buf, lg in pool.imap(_work, frames, chunksize=4):
            proc.stdin.write(buf)
            logs.append(lg)
            if fi % 120 == 0:
                print(f"  frame {fi}/{NFRAMES}", flush=True)
    proc.stdin.close()
    proc.wait()
    json.dump(logs, open(os.path.join(HERE, "data", "frame_log.json"), "w"))
    return proc.returncode


def render_stills(frames, outdir):
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    r = Renderer()
    for fi in frames:
        arr = r.frame(fi)
        Image.fromarray(arr[..., :3]).save(os.path.join(outdir, f"f{fi:04d}.png"))
        print("still", fi, r.log.get("readout", ""))


if __name__ == "__main__":
    if sys.argv[1] == "stills":
        render_stills([int(x) for x in sys.argv[2:]], os.path.join(HERE, "stills"))
    elif sys.argv[1] == "video":
        sys.exit(render_video(sys.argv[2]))
