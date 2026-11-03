"""SimTheory #03 - frame renderer.

Pure numpy/Pillow, additive (light-adding) compositing on a pure black HDR buffer, analytic anti-aliased
shapes (signed distance), multi-scale bloom. Every pendulum angle is evaluated from the RK4 trajectories in sim.py.

  python3 render.py --still 0 14.15 ...     -> build/stills/*.png
  python3 render.py --video                 -> build/video.mp4 (silent, 1440x2560@60)
"""
import argparse
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import sim
import timeline as TL

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
W, H = 1440, 2560

# --- safe area (clear of top 15 %, bottom 35 %, right 20 %) ---------------------------------
SAFE_X1 = int(W * 0.80)          # 1152
SAFE_Y0 = int(H * 0.15)          # 384
SAFE_Y1 = int(H * 0.65)          # 1664
CX = 576                         # content centre line (centre of 0..1152)

# --- series palette (SimTheory) ----------------------------------------------------------
CYAN = np.array([0.10, 0.92, 1.00], np.float32)
ORANGE = np.array([1.00, 0.50, 0.06], np.float32)
PURPLE = np.array([0.70, 0.30, 1.00], np.float32)
WHITE = np.array([1.0, 1.0, 1.0], np.float32)
DIVIDER = np.array([0.42, 0.48, 0.56], np.float32)

FONT_DIR = "/usr/share/fonts/opentype/inter"
F_CAPTION = os.path.join(FONT_DIR, "Inter-ExtraBold.otf")
F_LABEL = os.path.join(FONT_DIR, "Inter-Bold.otf")
F_COUNTER = os.path.join(FONT_DIR, "Inter-Black.otf")
F_TAG = os.path.join(FONT_DIR, "Inter-Light.otf")
if not os.path.exists(F_COUNTER):
    F_COUNTER = os.path.join(FONT_DIR, "Inter-ExtraBold.otf")

# --- layout (px) ---------------------------------------------------------------------------
CAP_TOP = 520                    # centre of the first caption line (block grows downward)
CAP_SIZE = 84
CAP_MAXW = 1010
TAG_POS = (60, 404)
LABEL_Y = 772
COUNTER_Y = 886
PIVOT_Y = 1000
PIV_E = (312.0, PIVOT_Y)
PIV_H = (840.0, PIVOT_Y)
PIV_G = (float(CX), PIVOT_Y)
LPX = 530.0                      # 1 m on screen
ROD_R = 5.0                      # rod half-width -> 10 px rod
BOB_R = 36.0
FORMULA_C = (float(CX), 1268.0)
FORMULA_W = 930.0                # on-screen ink width of the formula
FIELD_Y = (730.0, 1655.0)

# ---------------------------------------------------------------------------------------------
E = sim.Traj(sim.G_EARTH, 13.0)
HV = sim.Traj(sim.G_HEAVY, 13.0)
GH = sim.Traj(sim.G_GHOST, 7.0)
TH0 = sim.THETA0
EARTH_SYNC_TH = float(E.theta(TL.SYNC_SIM))


def theta_e(t):
    """Displayed Earth angle (rad) at video time t (array ok)."""
    t = np.asarray(t, float)
    tau = np.clip(t - TL.T_REL, 0.0, TL.SYNC_SIM)
    th = E.theta(tau)
    # held after the freeze: the 0.03 deg gap to the release angle (Earth reached it 16.8 ms earlier)
    # is closed invisibly while frozen, so the loop lands exactly on frame 1
    corr = np.clip((t - (TL.T_SYNC + 2.0)) / 4.0, 0, 1)
    corr = corr * corr * (3 - 2 * corr)
    return np.where(t > TL.T_SYNC, th + (TH0 - EARTH_SYNC_TH) * corr, th)


def theta_h(t):
    t = np.asarray(t, float)
    return HV.theta(np.clip(t - TL.T_REL, 0.0, TL.SYNC_SIM))


def theta_g(t):
    t = np.asarray(t, float)
    return GH.theta(np.clip(t - TL.GHOST_REL, 0.0, 6.9))


def bob_xy(piv, th):
    return piv[0] + LPX * np.sin(th), piv[1] + LPX * np.cos(th)


# =============================================================================================
class Frame:
    def __init__(self, zoom=1.0, zc=FORMULA_C):
        self.core = np.zeros((H, W, 3), np.float32)
        self.glow = np.zeros((H, W, 3), np.float32)
        self.z = zoom
        self.zc = zc

    # camera (world -> screen). Captions / tag are drawn in screen space (cam=False).
    def P(self, x, y):
        return self.zc[0] + (x - self.zc[0]) * self.z, self.zc[1] + (y - self.zc[1]) * self.z

    def _add(self, cov, x0, y0, color, alpha, glow_w):
        h, w = cov.shape
        if h <= 0 or w <= 0:
            return
        c = cov[..., None] * (color * alpha)
        self.core[y0:y0 + h, x0:x0 + w] += c
        if glow_w:
            self.glow[y0:y0 + h, x0:x0 + w] += c * glow_w

    # ---- analytic primitives (screen space) ------------------------------------------
    @staticmethod
    def _bbox(xs, ys, pad):
        x0 = max(int(math.floor(min(xs) - pad)), 0)
        x1 = min(int(math.ceil(max(xs) + pad)) + 1, W)
        y0 = max(int(math.floor(min(ys) - pad)), 0)
        y1 = min(int(math.ceil(max(ys) + pad)) + 1, H)
        return x0, x1, y0, y1

    def capsule_cov(self, ax, ay, bx, by, r, bbox=None):
        x0, x1, y0, y1 = bbox or self._bbox((ax, bx), (ay, by), r + 2)
        if x1 <= x0 or y1 <= y0:
            return None, 0, 0
        px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
        py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        if ll < 1e-9:
            d = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        else:
            u = np.clip(((px - ax) * dx + (py - ay) * dy) / ll, 0, 1)
            d = np.sqrt((px - ax - u * dx) ** 2 + (py - ay - u * dy) ** 2)
        return np.clip(r + 0.5 - d, 0, 1), x0, y0

    def capsule(self, a, b, r, color, alpha=1.0, glow_w=0.0, cam=True):
        if cam:
            a, b, r = self.P(*a), self.P(*b), r * self.z
        cov, x0, y0 = self.capsule_cov(a[0], a[1], b[0], b[1], r)
        if cov is not None:
            self._add(cov, x0, y0, color, alpha, glow_w)

    def disc(self, c, r, color, alpha=1.0, glow_w=0.0, cam=True):
        self.capsule(c, c, r, color, alpha, glow_w, cam)

    def ring(self, c, R, width, color, alpha=1.0, glow_w=0.0, cam=True):
        if cam:
            c, R, width = self.P(*c), R * self.z, width * self.z
        x0, x1, y0, y1 = self._bbox((c[0],), (c[1],), R + width + 2)
        if x1 <= x0 or y1 <= y0:
            return
        px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
        py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        d = np.abs(np.sqrt((px - c[0]) ** 2 + (py - c[1]) ** 2) - R)
        self._add(np.clip(width / 2 + 0.5 - d, 0, 1), x0, y0, color, alpha, glow_w)

    def dot(self, c, sigma, color, alpha, glow_w=0.0, cam=True):
        """soft gaussian particle"""
        if cam:
            c, sigma = self.P(*c), sigma * self.z
        x0, x1, y0, y1 = self._bbox((c[0],), (c[1],), 3 * sigma + 1)
        if x1 <= x0 or y1 <= y0:
            return
        px = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
        py = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        cov = np.exp(-((px - c[0]) ** 2 + (py - c[1]) ** 2) / (2 * sigma * sigma))
        self._add(cov, x0, y0, color, alpha, glow_w)

    def polyline_max(self, pts, radii, alphas, color, glow_w, cam=True):
        """fading trail: per-segment coverage combined with max() so joints don't bead up."""
        if len(pts) < 2:
            return
        pts = np.asarray(pts, float)
        if cam:
            pts = np.stack(self.P(pts[:, 0], pts[:, 1]), 1)
            radii = np.asarray(radii) * self.z
        rmax = float(np.max(radii))
        x0, x1, y0, y1 = self._bbox(pts[:, 0], pts[:, 1], rmax + 2)
        if x1 <= x0 or y1 <= y0:
            return
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            r = 0.5 * (radii[i] + radii[i + 1])
            al = 0.5 * (alphas[i] + alphas[i + 1])
            if al <= 0.003:
                continue
            sb = self._bbox((ax, bx), (ay, by), r + 2)
            sb = (max(sb[0], x0), min(sb[1], x1), max(sb[2], y0), min(sb[3], y1))
            cov, sx, sy = self.capsule_cov(ax, ay, bx, by, r, sb)
            if cov is None:
                continue
            sub = acc[sy - y0:sy - y0 + cov.shape[0], sx - x0:sx - x0 + cov.shape[1]]
            np.maximum(sub, cov * al, out=sub)
        self._add(acc, x0, y0, color, 1.0, glow_w)

    def sprite(self, mask, cx, cy, scale, color, alpha=1.0, glow_w=0.0, cam=True):
        """mask: float32 [0,1] image, placed with its centre at (cx, cy), uniformly scaled (sub-pixel exact)."""
        if alpha <= 0.002:
            return
        if cam:
            (cx, cy), scale = self.P(cx, cy), scale * self.z
        mh, mw = mask.shape
        hw, hh = mw * scale / 2, mh * scale / 2
        x0, x1 = int(math.floor(cx - hw)) - 1, int(math.ceil(cx + hw)) + 1
        y0, y1 = int(math.floor(cy - hh)) - 1, int(math.ceil(cy + hh)) + 1
        if abs(scale - 1) < 1e-6 and abs((cx - hw) - round(cx - hw)) < 1e-6 and abs((cy - hh) - round(cy - hh)) < 1e-6:
            sx, sy = int(round(cx - hw)), int(round(cy - hh))
            cov = mask
        else:
            # dest pixel (i, j) -> source ((x0 + i + .5 - (cx - hw)) / scale - .5)
            a = 1.0 / scale
            c0 = (x0 - (cx - hw)) / scale + 0.5 * a - 0.5
            f0 = (y0 - (cy - hh)) / scale + 0.5 * a - 0.5
            img = Image.fromarray(mask, "F")
            out = img.transform((x1 - x0, y1 - y0), Image.AFFINE, (a, 0, c0 + 0.5, 0, a, f0 + 0.5),
                                resample=Image.BICUBIC)
            cov = np.clip(np.asarray(out, np.float32), 0, 1)
            sx, sy = x0, y0
        # clip to frame
        cx0, cy0 = max(sx, 0), max(sy, 0)
        cx1, cy1 = min(sx + cov.shape[1], W), min(sy + cov.shape[0], H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        self._add(cov[cy0 - sy:cy1 - sy, cx0 - sx:cx1 - sx], cx0, cy0, color, alpha, glow_w)

    # ---- final -------------------------------------------------------------------------
    def finish(self, frame_idx=0):
        g = self.glow
        g2 = g.reshape(H // 2, 2, W // 2, 2, 3).mean((1, 3))
        g4 = g2.reshape(H // 4, 2, W // 4, 2, 3).mean((1, 3))
        g8 = g4.reshape(H // 8, 2, W // 8, 2, 3).mean((1, 3))
        b1 = ndimage.gaussian_filter(g2, (5, 5, 0))          # ~10 px
        b2 = ndimage.gaussian_filter(g4, (7, 7, 0))          # ~28 px
        b3 = ndimage.gaussian_filter(g8, (11, 11, 0))        # ~90 px
        acc = up2(b2 * 1.0 + up2(b3) * 1.3) + b1 * 1.1      # at half res
        out = self.core + up2(acc)
        # neon tone: channels clip, overflow spills to white (hot cores)
        m = out.max(axis=2, keepdims=True)
        spill = np.clip((m - 1.0) * 0.45, 0, 1)
        out = np.clip(out, 0, 1) * (1 - spill) + spill
        rng = np.random.default_rng(1000 + frame_idx)
        dither = rng.random((H, W, 1), dtype=np.float32)
        return np.clip(out * 255.0 + dither, 0, 255).astype(np.uint8)


def up2(a):
    h, w, _ = a.shape
    out = np.empty((h * 2, w * 2, 3), np.float32)
    for c in range(3):
        out[..., c] = np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c]), "F").resize((w * 2, h * 2), Image.BILINEAR))
    return out


# ---- text ------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=None)
def text_mask(text, path, size, tracking=0, maxw=0, line_h=1.12):
    """White text -> float32 coverage mask (multi-line centred, optional word wrap to maxw)."""
    f = font(path, size)

    def width(s):
        if tracking == 0:
            return f.getlength(s)
        return sum(f.getlength(ch) for ch in s) + tracking * (len(s) - 1)

    lines = [text]
    if maxw and width(text) > maxw:
        words = text.split(" ")
        best = None
        for k in range(1, len(words)):             # most balanced 2-line split that fits
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wmax = max(width(a), width(b))
            if wmax <= maxw and (best is None or wmax < best[0]):
                best = (wmax, [a, b])
        lines = best[1] if best else [text]
    asc, desc = f.getmetrics()
    lh = int(round(size * line_h))
    pad = int(size * 0.3)
    tw = int(max(width(s) for s in lines)) + 2 * pad
    th = lh * (len(lines) - 1) + asc + desc + 2 * pad
    img = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        x = (tw - width(s)) / 2
        y = pad + i * lh
        if tracking == 0:
            d.text((x, y), s, font=f, fill=255)
        else:
            for ch in s:
                d.text((x, y), ch, font=f, fill=255)
                x += f.getlength(ch) + tracking
    a = np.asarray(img, np.float32) / 255.0
    # trim to ink box vertically keeping centre on the cap-height middle
    return a, lines


@lru_cache(maxsize=None)
def formula_masks():
    d = os.path.join(BUILD, "formula")
    out = {}
    for n in ["all", "T", "eq", "twopi", "root", "L", "slash", "g"]:
        out[n] = np.asarray(Image.open(os.path.join(d, n + ".png")), np.float32)[..., 3] / 255.0
    ys, xs = np.nonzero(out["all"] > 0.03)
    box = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
    pad = 6
    x0, x1, y0, y1 = box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad
    for n in out:
        out[n] = np.ascontiguousarray(out[n][y0:y1, x0:x1])
    return out


# ---- scene pieces --------------------------------------------------------------------------
_rng = np.random.default_rng(3)


def _field(n, x_lo, x_hi, wraps):
    xs = _rng.uniform(x_lo, x_hi, n)
    y0 = _rng.uniform(0, 1, n)
    k = _rng.choice(wraps, n)
    b = _rng.uniform(0.45, 1.0, n)
    s = _rng.uniform(2.6, 3.6, n)
    return xs, y0, k, b, s


# gravity field: speeds are integer numbers of wraps per loop (DUR) so the field loops seamlessly
FIELD_E = _field(30, 70, 545, [3, 4, 4, 5])
FIELD_H = _field(78, 607, 1100, [14, 16, 17, 18, 20])


def draw_field(fr, t, field, color, alpha, streak_s):
    xs, y0, k, b, s = field
    hgt = FIELD_Y[1] - FIELD_Y[0]
    v = k * hgt / TL.DUR                          # px/s
    ph = (y0 + k * (t / TL.DUR)) % 1.0
    ys = FIELD_Y[0] + ph * hgt
    edge = np.clip(np.minimum(ph, 1 - ph) / 0.12, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    for i in range(len(xs)):
        a = alpha * b[i] * edge[i]
        if a < 0.004:
            continue
        ln = v[i] * streak_s
        fr.capsule((xs[i], ys[i] - ln), (xs[i], ys[i]), s[i], color, a, glow_w=0.9)


# glow sparks shed by the bobs; they fall with (a cosmetic multiple of) each side's gravity
SPARK_RATE = 34.0
SPARK_LIFE = 0.85


def _sparks(seed):
    n = int(TL.DUR * SPARK_RATE)
    r = np.random.default_rng(seed)
    ang = r.uniform(0, 2 * math.pi, n)
    sp = r.uniform(15, 75, n)
    return (np.arange(n) / SPARK_RATE, np.cos(ang) * sp, np.sin(ang) * sp - 20, r.uniform(0.35, 1.0, n),
            r.uniform(2.4, 4.2, n), r.uniform(-14, 14, (n, 2)))


SPARKS_E = _sparks(11)
SPARKS_H = _sparks(12)


def draw_sparks(fr, t, sparks, bobfn, color, acc, alpha):
    ts, vx, vy, br, sg, off = sparks
    for wrap in (0.0, -TL.DUR):                      # periodic in DUR -> seamless loop
        age = t - (ts + wrap)
        sel = np.nonzero((age >= 0) & (age < SPARK_LIFE))[0]
        if len(sel) == 0:
            continue
        bx, by = bobfn(ts[sel])                      # bob position when the spark was shed
        for j, i in enumerate(sel):
            a = age[i]
            x = bx[j] + off[i, 0] + vx[i] * a
            y = by[j] + off[i, 1] + vy[i] * a + 0.5 * acc * a * a
            f = (1 - a / SPARK_LIFE)
            fr.dot((x, y), sg[i], color, alpha * br[i] * f * f * 0.55, glow_w=1.2)


def pop_scale(t, events):
    """counter pop: quick swell and settle after each count event"""
    s = 0.0
    for te in events:
        u = t - te
        if 0 <= u < 0.4:
            s += 0.30 * (u / 0.045) * math.exp(1 - u / 0.045)
    return 1.0 + s


def caption_state(t):
    """list of (text, alpha, dy) currently visible"""
    out = []
    for text, a, b in TL.CAPTIONS:
        if t < a or t >= b + TL.CAP_FADE_OUT:
            continue
        if a > 0 and t < a + TL.CAP_FADE_IN:
            u = TL.smooth((t - a) / TL.CAP_FADE_IN)
            out.append((text, u, (1 - u) * 18))
        elif t >= b:
            u = TL.smooth((t - b) / TL.CAP_FADE_OUT)
            out.append((text, 1 - u, -u * 12))
        else:
            out.append((text, 1.0, 0.0))
    return out


def draw_pendulum(fr, t, piv, thfn, color, dim, shutter, trail_s, trail_fade, mount_alpha=1.0, max_sub=8):
    # motion blur (symmetric shutter) - samples of the true trajectory within the frame interval
    span = shutter / TL.FPS
    th_lo, th_hi = thfn(np.array([t - span / 2, t + span / 2]))
    dpx = abs(th_hi - th_lo) * LPX
    n = int(min(max_sub, max(1, math.ceil(dpx / 3.0))))
    ts = t + (np.arange(n) / max(n - 1, 1) - 0.5) * span if n > 1 else np.array([t])
    ths = thfn(ts)
    # mount
    fr.capsule((piv[0] - 46, piv[1]), (piv[0] + 46, piv[1]), 4.5, color, 0.55 * dim * mount_alpha, glow_w=0.4)
    # trail
    if trail_s > 0 and trail_fade > 0:
        tt = t - np.linspace(0, trail_s, 70)
        bx, by = bob_xy(piv, thfn(tt))
        age = np.linspace(0, 1, 70)
        alph = (1 - age) ** 1.6 * 0.75 * trail_fade * dim
        rad = 7.0 * (1 - age) + 1.6
        if np.hypot(bx[0] - bx[-1], by[0] - by[-1]) > 1.0:
            fr.polyline_max(np.stack([bx, by], 1), rad, alph, color, glow_w=0.9)
    for th in ths:
        bx, by = bob_xy(piv, th)
        a = dim / n
        fr.capsule(piv, (bx, by), ROD_R, color, 0.85 * a, glow_w=0.55)
        fr.disc((bx, by), BOB_R, color, 1.25 * a, glow_w=1.4)
        fr.disc((bx, by), BOB_R * 0.48, WHITE, 0.55 * a, glow_w=0.3)
    fr.disc(piv, 9, color, 1.1 * dim * mount_alpha, glow_w=0.8)
    fr.disc(piv, 4, WHITE, 0.6 * dim * mount_alpha)


def render(fi):
    t = fi / TL.FPS
    fr = Frame(zoom=TL.cam_zoom(t))
    frozen = t >= TL.T_SYNC
    since_sync = t - TL.T_SYNC

    # pendulum dimming during reveal / punchline
    dim = 1.0 - 0.80 * TL.ramp(t, TL.T_FORMULA - 0.3, TL.T_FORMULA + 0.4) + 0.80 * TL.ramp(t, 26.6, 27.6)
    if t >= TL.GHOST_IN[0] - 0.2 and t < 26.6:
        dim = min(dim, 0.20)
    field_a = 0.85 - 0.45 * TL.ramp(t, TL.T_FORMULA - 0.3, TL.T_FORMULA + 0.4) + 0.45 * TL.ramp(t, 26.6, 27.6)

    # gravity field (behind)
    draw_field(fr, t, FIELD_E, CYAN, 0.30 * field_a, 0.035)
    draw_field(fr, t, FIELD_H, ORANGE, 0.30 * field_a, 0.035)

    # divider
    top, bot = 725.0, 1655.0
    for k in range(24):
        y_a = top + (bot - top) * k / 24
        y_b = top + (bot - top) * (k + 1) / 24
        u = (k + 0.5) / 24
        a = math.sin(math.pi * u) ** 0.8
        fr.capsule((CX, y_a), (CX, y_b), 2.2, DIVIDER, 0.42 * a, glow_w=0.25)

    # sparks
    def bobE(ts):
        return bob_xy(PIV_E, theta_e(ts))

    def bobH(ts):
        return bob_xy(PIV_H, theta_h(ts))

    draw_sparks(fr, t, SPARKS_E, bobE, CYAN, 45.0, dim)
    draw_sparks(fr, t, SPARKS_H, bobH, ORANGE, 450.0, dim)

    # trails: frozen at the sync point and dissolving
    trail_fade = 1.0 if not frozen else max(0.0, 1 - since_sync / 0.7)
    ts_e = min(t, TL.T_SYNC)
    # draw pendulums (time is frozen after the sync -> evaluate at clamped time)
    tE = min(t, TL.T_SYNC) if frozen else t
    draw_pendulum(fr, t if not frozen else t, PIV_E, theta_e, CYAN, dim, 0.5, 0.55, trail_fade)
    draw_pendulum(fr, t, PIV_H, theta_h, ORANGE, dim, 0.5, 0.30, trail_fade)

    # little ring at each full swing, a bigger one at the sync
    for evs, piv, thfn, col in ((TL.CNT_E, PIV_E, theta_e, CYAN), (TL.CNT_H, PIV_H, theta_h, ORANGE)):
        for i, te in enumerate(evs):
            u = t - te
            last = (i == len(evs) - 1)
            dur = 0.9 if last else 0.38
            if 0 <= u < dur:
                v = u / dur
                c = bob_xy(piv, thfn(te))
                R = BOB_R + (76 if last else 46) * (1 - (1 - v) ** 2.2)
                fr.ring(c, R, 7 if last else 5, col, (1 - v) ** 1.5 * (1.0 if last else 0.55), glow_w=1.0)

    # ghost (100x gravity): real sim, 360-degree shutter, many samples -> blur
    ga = TL.ghost_alpha(t)
    if ga > 0:
        shim = 0.92 + 0.08 * math.sin(fi * 2.1) * math.sin(fi * 0.77)
        a = TL.GHOST_ALPHA * ga * shim
        span = 1.0 / TL.FPS
        n = 28
        ts = t + (np.arange(n) / (n - 1) - 0.5) * span
        ths = theta_g(ts)
        fr.capsule((PIV_G[0] - 46, PIV_G[1]), (PIV_G[0] + 46, PIV_G[1]), 4.5, PURPLE, 0.5 * a, glow_w=0.5)
        for th in ths:
            bx, by = bob_xy(PIV_G, th)
            fr.capsule(PIV_G, (bx, by), ROD_R, PURPLE, 1.0 * a / n * 2.2, glow_w=0.8)
            fr.disc((bx, by), BOB_R, PURPLE, 1.3 * a / n * 2.2, glow_w=1.6)
        # long-exposure persistence over one full ghost period -> the whole arc smears into a blur
        nl = 36
        tl = t - np.arange(1, nl + 1) / nl * sim.exact_period(sim.G_GHOST)
        for th in theta_g(tl):
            bx, by = bob_xy(PIV_G, th)
            fr.capsule(PIV_G, (bx, by), ROD_R, PURPLE, 0.55 * a / nl, glow_w=0.8)
            fr.disc((bx, by), BOB_R, PURPLE, 0.9 * a / nl, glow_w=1.2)
        fr.disc(PIV_G, 9, PURPLE, a, glow_w=0.8)
        m, _ = text_mask("100× GRAVITY", F_LABEL, 46, 3)
        fr.sprite(m, CX, 1612, 1.0, PURPLE, 1.15 * a, glow_w=0.5)

    # labels + counters (world space)
    lab_a = 1.0 - 0.35 * TL.ramp(t, TL.GHOST_IN[0] - 0.2, TL.GHOST_IN[0] + 0.3) + 0.35 * TL.ramp(t, 26.6, 27.4)
    for text, x, col in (("EARTH · 1g", PIV_E[0], CYAN), ("10× GRAVITY", PIV_H[0], ORANGE)):
        m, _ = text_mask(text, F_LABEL, 46, 3)
        fr.sprite(m, x, LABEL_Y, 1.0, col, 1.1 * lab_a, glow_w=0.45)
    for evs, x, col in ((TL.CNT_E, PIV_E[0], CYAN), (TL.CNT_H, PIV_H[0], ORANGE)):
        if t < TL.T_RESET:
            n = int(np.sum(evs <= t + 1e-9))
            ps = pop_scale(t, evs)
            flash = (ps - 1) * 1.1
        else:
            n = 0
            ps = pop_scale(t, [TL.T_RESET])
            flash = (ps - 1) * 1.1
        m, _ = text_mask(str(n), F_COUNTER, 150)
        fr.sprite(m, x, COUNTER_Y, ps, col, (1.15 + flash) * lab_a, glow_w=0.7 + flash)

    # formula
    if TL.T_FORMULA <= t < TL.T_FORMULA_OUT[1]:
        fm = formula_masks()
        sc = FORMULA_W / fm["all"].shape[1]
        out_a = 1 - TL.ramp(t, *TL.T_FORMULA_OUT)
        for name, dt in TL.FORMULA_GROUPS:
            u = (t - TL.T_FORMULA - dt) / 0.38
            if u <= 0:
                continue
            e = TL.smoother(u)
            col = ORANGE if name == "g" else WHITE
            a = e * out_a
            dy = (1 - e) * 26
            s = sc * (0.94 + 0.06 * e)
            glow = 0.25
            if name == "root":
                col = np.array([0.80, 1.0, 1.0], np.float32)
                if t >= TL.T_ROOT_GLOW:
                    ph = (t - TL.T_ROOT_GLOW)
                    pulse = 0.5 - 0.5 * math.cos(2 * math.pi * ph / 0.9)
                    rise = TL.smooth(ph / 0.5)
                    glow = 0.25 + rise * (1.4 + 1.8 * pulse)
                    a = a * (1.0 + rise * 0.35 * pulse)
                    # the pulse also tints the glow cyan
                    fr.sprite(fm["root"], FORMULA_C[0], FORMULA_C[1] + dy, s, CYAN, rise * (0.4 + 0.9 * pulse) * out_a,
                              glow_w=2.2)
            fr.sprite(fm[name], FORMULA_C[0], FORMULA_C[1] + dy, s, col, a, glow_w=glow)

    # ---- screen space: series tag + captions --------------------------------------------
    m, _ = text_mask("SIM #03", F_TAG, 48, 6)
    fr.sprite(m, TAG_POS[0] + m.shape[1] / 2 - 13, TAG_POS[1] + m.shape[0] / 2 - 13, 1.0, CYAN, 1.0, glow_w=0.35,
              cam=False)
    for text, a, dy in caption_state(t):
        m, lines = text_mask(text, F_CAPTION, CAP_SIZE, 0, CAP_MAXW)
        cy = CAP_TOP + (len(lines) - 1) * round(CAP_SIZE * 1.12) / 2
        fr.sprite(m, CX, cy + dy, 1.0, WHITE, a, glow_w=0.16, cam=False)
    return fr.finish(fi)


# =============================================================================================
def _work(fi):
    return render(fi).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs="*", type=float)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(BUILD, "video.mp4"))
    ap.add_argument("--range", nargs=2, type=int)
    args = ap.parse_args()
    if args.still is not None:
        os.makedirs(os.path.join(BUILD, "stills"), exist_ok=True)
        for ts in args.still:
            fi = int(round(ts * TL.FPS))
            p = os.path.join(BUILD, "stills", f"f{fi:04d}.png")
            Image.fromarray(render(fi)).save(p)
            print(p)
    if args.video:
        import multiprocessing as mp
        a, b = args.range or (0, TL.NFRAMES)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(TL.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "10",
               "-profile:v", "high", "-level", "5.1", "-pix_fmt", "yuv420p",
               "-x264-params", "keyint=60:min-keyint=60:bframes=3:aq-mode=3",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
               args.out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with mp.get_context("fork").Pool(args.jobs) as pool:
            for k, buf in enumerate(pool.imap(_work, range(a, b), chunksize=2)):
                ff.stdin.write(buf)
                if k % 60 == 0:
                    print(f"frame {a + k}/{b}", flush=True)
        ff.stdin.close()
        ff.wait()
        print("wrote", args.out)


if __name__ == "__main__":
    main()
