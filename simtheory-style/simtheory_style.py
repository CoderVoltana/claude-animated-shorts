"""SimTheory style kit: one file that draws, glows, captions and encodes every episode the same way.

Attach this file to every new chat and say:
    "Use the attached simtheory_style.py for all drawing, glow, text and encoding."

Rules for whoever (human or model) builds an episode with it:
  * Never type a hex code or font name in episode code. Use COLORS / ROLES / role_color() / TEXT styles.
  * Colours carry the same meaning in every episode (see ROLES).
  * Draw with Frame(...) and finish() it. Export only with VideoWriter or encode_frames().
  * Keep anything that must be read inside SAFE["square"] (it survives YouTube, the Instagram
    square / 4:5 / 3:4 crops and the Reels UI).

Needs: Python 3.9+, numpy, Pillow, ffmpeg on PATH. Fonts are bundled in ./fonts next to this file.

Quick test (writes a style test frame and a 3 s clip):
    python simtheory_style.py demo out_dir
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

VERSION = "1.0.0"

# --------------------------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------------------------
W, H, FPS = 1080, 1920, 60           # vertical 9:16, 60 fps for every platform

# --------------------------------------------------------------------------------------------
# Colours: exact codes. Episode code uses the NAMES, never the hex values.
# --------------------------------------------------------------------------------------------
COLORS = {
    "bg":       "#000000",   # deep black background
    "nebula":   "#120E33",   # very faint backdrop glow behind the subject
    "white":    "#F4F7FF",   # reference / true result
    "cyan":     "#22E5FF",   # main / ideal case
    "orange":   "#FF8A1E",   # perturbed / "different" case
    "purple":   "#A848FF",   # extra case 1
    "blue":     "#3A7BFF",   # extra case 2
    "magenta":  "#FF3FA4",   # extra case 3
    "text":     "#F4F7FF",   # captions
    "text_dim": "#A9BCDD",   # secondary captions
    "hud":      "#5AA0FF",   # graph axes, gridlines, labels
}

# One fixed meaning per colour for the whole series. Viewers learn it; no labels needed.
ROLES = {
    "reference": "white",    # the true / reference result
    "main":      "cyan",     # the main or ideal case
    "perturbed": "orange",   # the perturbed or "different" case
    "extra":     ["purple", "blue", "magenta"],   # additional cases, in this order
}
# Large crowds (ensembles, particle clouds) use this gradient at low opacity, behind the role colours.
ENSEMBLE_STOPS = ["cyan", "blue", "purple"]

# --------------------------------------------------------------------------------------------
# Glow: a single bloom recipe. Radii are in pixels at 1080x1920.
# --------------------------------------------------------------------------------------------
GLOW = {
    "radii":    (5, 16, 44),        # three bloom scales: tight halo, soft glow, wide atmosphere
    "weights":  (0.60, 0.42, 0.26),
    "strength": 1.0,                # overall bloom gain (keep at 1.0; only lower for very dense scenes)
    "knee":     0.70,               # highlights above this roll off softly instead of clipping
    "hue_keep": 0.85,               # 1.0 = overlaps stay fully saturated, 0.0 = overlaps go white
    "dither":   True,               # +-0.5 LSB noise before 8-bit: no banding in dark gradients
}

# --------------------------------------------------------------------------------------------
# Sizes in pixels at 1080x1920
# --------------------------------------------------------------------------------------------
SIZES = {
    "line":           3.2,    # rods, links, solid curves
    "trail_head":     5.0,    # trail width at the newest point
    "trail_tail":     1.2,    # trail width at the oldest point
    "trail_chunks":   12,     # alpha/width steps along a trail
    "bob":            9.0,    # bob / particle core radius
    "pivot":          6.0,    # fixed hinge radius
    "ensemble_line":  1.2,    # background crowd lines
    "ensemble_alpha": 0.16,   # background crowd opacity
    "spark":          2.5,    # particle size
}

# --------------------------------------------------------------------------------------------
# Type
# --------------------------------------------------------------------------------------------
FONT_FILES = {
    "display": "InterDisplay-Black.otf",
    "body":    "Inter-SemiBold.otf",
    "mono":    "JetBrainsMono-Bold.ttf",
    "mono_md": "JetBrainsMono-Medium.ttf",
}
TEXT = {   # font, size px, tracking px, colour, glow colour (None = no glow)
    "headline": dict(font="display", size=104, tracking=3,  color="white",    glow="cyan"),
    "title":    dict(font="display", size=180, tracking=14, color="white",    glow="purple"),
    "subhead":  dict(font="body",    size=50,  tracking=0,  color="text",     glow=None),
    "caption":  dict(font="body",    size=42,  tracking=0,  color="text_dim", glow=None),
    "label":    dict(font="mono",    size=30,  tracking=1,  color="text",     glow=None),
    "hud":      dict(font="mono_md", size=24,  tracking=1,  color="hud",      glow=None),
    "number":   dict(font="mono",    size=44,  tracking=0,  color="cyan",     glow="cyan"),
}
GRADIENT_TITLE = ["cyan", "blue", "purple", "magenta", "orange"]   # only for the one big reveal word

# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------
SAFE = {
    "square":   (0, 420, 1080, 1500),   # x0, y0, x1, y1: everything that must be read goes in here
    "margin":   64,                     # side margin inside the square
    "text_y":   505,                    # centre line of the first caption line
    "subject":  (540, 1040),            # centre of the main visual
    "hud_band": (1320, 1500),           # graphs / legends / readouts
}
TAG = {
    "format": "SIM #{:02d}",            # the series tag, e.g. SIM #04
    "xy":     (64, 448),                # top-left corner, inside the square-safe zone
    "style":  "hud",
    "color":  "cyan",
}

# --------------------------------------------------------------------------------------------
# Export: one colour standard for every file (BT.709, limited range), 60 fps, -14 LUFS
# --------------------------------------------------------------------------------------------
EXPORT = {
    "fps": FPS, "crf": 15, "preset": "slow", "pix_fmt": "yuv420p", "profile": "high",
    "matrix": "bt709", "range": "tv",
    "lufs": -14, "true_peak": -1.5, "aac_kbps": 256, "audio_rate": 48000,
}


# ============================================================================================
# helpers
# ============================================================================================
def hex_rgb(h: str) -> np.ndarray:
    """'#22E5FF' -> float32 array in 0..1."""
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def C(name: str) -> np.ndarray:
    """Colour by name (from COLORS) as a float array."""
    return hex_rgb(COLORS[name])


def role_color(role: str, index: int = 0) -> np.ndarray:
    """Colour for a role: 'reference', 'main', 'perturbed', or 'extra' (index 0..2)."""
    r = ROLES[role]
    return C(r[index % len(r)] if isinstance(r, list) else r)


def ensemble_color(u: float) -> np.ndarray:
    """Colour along the background-crowd gradient, u in 0..1."""
    stops = [C(n) for n in ENSEMBLE_STOPS]
    u = min(max(u, 0.0), 1.0) * (len(stops) - 1)
    k = min(int(u), len(stops) - 2)
    return stops[k] * (1 - (u - k)) + stops[k + 1] * (u - k)


_FONT_CACHE: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """Bundled font (./fonts next to this file); falls back to system copies, and warns loudly if missing."""
    key = (kind, size)
    if key not in _FONT_CACHE:
        name = FONT_FILES[kind]
        here = Path(__file__).resolve().parent / "fonts" / name
        cands = [here, Path("fonts") / name,
                 Path("/usr/share/fonts/opentype/inter") / name,
                 Path("/usr/share/fonts/truetype/jetbrains-mono") / name]
        for p in cands:
            if p.exists():
                _FONT_CACHE[key] = ImageFont.truetype(str(p), size)
                break
        else:
            raise FileNotFoundError(
                f"simtheory_style: font {name} not found. Put the kit's fonts/ folder next to "
                f"simtheory_style.py (a fallback font would change the series look).")
    return _FONT_CACHE[key]


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(t):
    return 1 - (1 - np.clip(t, 0, 1)) ** 3


def _box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur (3 box passes per axis) of an HxW or HxWxC float array."""
    if sigma <= 0.3:
        return a
    r = max(1, int(round((math.sqrt(12 * sigma * sigma / 3 + 1) - 1) / 2)))
    for _ in range(3):
        a = _box_blur(_box_blur(a, r, 0), r, 1)
    return a


def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c]), "F").resize((w, h), Image.BILINEAR))
                     for c in range(a.shape[2])], -1)


# ============================================================================================
# Frame: additive light buffer + text overlay
# ============================================================================================
class Frame:
    """One video frame. Draw light with line/trail/bob/dots (additive, glows),
    then captions with text()/tag() (crisp, drawn after the glow), then finish() -> uint8 RGB."""

    SS = 3   # supersampling for anti-aliased geometry

    def __init__(self, w: int = W, h: int = H, index: int = 0, nebula_center=None):
        self.w, self.h, self.index = w, h, index
        self.light = np.zeros((h, w, 3), np.float32)      # everything that glows
        self.overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.glow_layer = np.zeros((h, w, 3), np.float32)  # text glow, added after bloom
        self.nebula_center = nebula_center or SAFE["subject"]
        self.s = h / 1920.0                                 # pixel sizes scale with resolution

    # ---------------- geometry ----------------
    def _stamp(self, polylines, width, color, alpha, dots=None, caps=True):
        """Rasterise polylines (and optional dots) as one coverage mask, add color*alpha*coverage."""
        width = width * self.s
        pts_all = [np.asarray(p, np.float64) for p in polylines if len(p) >= 1]
        if dots is not None and len(dots):
            pts_all.append(np.asarray(dots, np.float64)[:, :2])
        if not pts_all:
            return
        allp = np.concatenate(pts_all)
        pad = width + 4 + (max(d[2] for d in dots) * self.s if dots is not None and len(dots) else 0)
        x0 = max(int(math.floor(allp[:, 0].min() - pad)), 0)
        y0 = max(int(math.floor(allp[:, 1].min() - pad)), 0)
        x1 = min(int(math.ceil(allp[:, 0].max() + pad)), self.w)
        y1 = min(int(math.ceil(allp[:, 1].max() + pad)), self.h)
        if x1 <= x0 or y1 <= y0:
            return
        S = self.SS
        im = Image.new("L", ((x1 - x0) * S, (y1 - y0) * S), 0)
        d = ImageDraw.Draw(im)
        wpx = max(1, int(round(width * S)))
        rad = wpx / 2
        for p in polylines:
            p = (np.asarray(p, np.float64) - (x0, y0)) * S
            if len(p) >= 2:
                d.line([tuple(q) for q in p], fill=255, width=wpx, joint="curve")
            for q in ((p[0], p[-1]) if caps else ()):    # round caps
                d.ellipse((q[0] - rad, q[1] - rad, q[0] + rad, q[1] + rad), fill=255)
        if dots is not None:
            for x, y, r in dots:
                r = r * self.s * S
                cx, cy = (x - x0) * S, (y - y0) * S
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        cov = np.asarray(im.resize((x1 - x0, y1 - y0), Image.BOX), np.float32) / 255.0
        self.light[y0:y1, x0:x1] += cov[..., None] * (np.asarray(color, np.float32) * alpha)

    def line(self, pts, color, width=None, alpha=1.0):
        """Solid polyline, e.g. a rod: pts = [(x, y), ...] in pixels."""
        self._stamp([pts], SIZES["line"] if width is None else width, color, alpha)

    def lines(self, polylines, color, width=None, alpha=None):
        """Many polylines in one colour (crowds). Overlaps inside one call don't stack."""
        self._stamp(polylines, SIZES["ensemble_line"] if width is None else width, color,
                    SIZES["ensemble_alpha"] if alpha is None else alpha)

    def trail(self, pts, color, alpha=1.0, head=None, tail=None, chunks=None, core=True):
        """Fading trail: pts ordered oldest -> newest. Width and opacity ramp up toward the head."""
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        head = SIZES["trail_head"] if head is None else head
        tail = SIZES["trail_tail"] if tail is None else tail
        n = chunks or SIZES["trail_chunks"]
        K = len(pts)
        for c in range(n):
            u = (c + 1) / n
            j0, j1 = int(c * (K - 1) / n), int((c + 1) * (K - 1) / n)
            if j1 <= j0:
                continue
            seg = pts[j0:j1 + 1]
            last = c == n - 1   # only the head gets a round cap: caps at chunk joins would double up
            self._stamp([seg], tail + (head - tail) * u, color, alpha * u * u, caps=last)
            if core and u > 0.6:   # hot white core near the head
                self._stamp([seg], 1.2, C("white"), alpha * 0.45 * (u - 0.6), caps=last)

    def blob(self, x, y, sigma, color, alpha=1.0):
        """Soft gaussian light (halo), sigma in px."""
        sg = sigma * self.s
        R = int(3 * sg) + 2
        x0, y0 = max(int(x) - R, 0), max(int(y) - R, 0)
        x1, y1 = min(int(x) + R + 1, self.w), min(int(y) + R + 1, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sg * sg))
        self.light[y0:y1, x0:x1] += g[..., None] * (np.asarray(color, np.float32) * alpha)

    def bob(self, x, y, color, r=None, alpha=1.0):
        """Glowing mass: soft coloured halo + solid core + white-hot centre."""
        r = SIZES["bob"] if r is None else r
        self.blob(x, y, r * 1.6, color, alpha * 0.55)
        self._stamp([], 0, color, alpha, dots=[(x, y, r)])
        self._stamp([], 0, C("white"), alpha * 0.85, dots=[(x, y, r * 0.42)])

    def pivot(self, x, y, alpha=1.0):
        self.bob(x, y, C("cyan") * 0.6 + C("white") * 0.4, r=SIZES["pivot"], alpha=alpha * 0.8)

    def dots(self, xy, color, r=None, alpha=1.0):
        """Many small particles in one colour: xy = [(x, y), ...]."""
        r = SIZES["spark"] if r is None else r
        self._stamp([], 0, color, alpha, dots=[(x, y, r) for x, y in xy])

    # ---------------- background ----------------
    _BG_CACHE: dict = {}

    def _background(self) -> np.ndarray:
        key = (self.w, self.h, tuple(self.nebula_center))
        if key not in Frame._BG_CACHE:
            bg = np.broadcast_to(C("bg"), (self.h, self.w, 3)).astype(np.float32).copy()
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            cx, cy = self.nebula_center
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (1100 * self.s)
            bg += C("nebula")[None, None] * (np.clip(1 - d, 0, 1) ** 1.6)[..., None] * 1.4
            Frame._BG_CACHE = {key: bg}
        return Frame._BG_CACHE[key]

    # ---------------- text ----------------
    def text(self, s, x, y, style="subhead", color=None, gradient=None, alpha=1.0, scale=1.0, anchor="mm",
             max_width=None):
        """Caption in a TEXT style. '\n' makes extra lines (stacked downward from y).
        Shrinks to fit max_width (default: the safe square minus margins).
        gradient = list of colour names left->right (only for the one big reveal word)."""
        st = TEXT[style]
        lines = s.split("\n")
        if len(lines) > 1:
            lh = st["size"] * scale * self.s * 1.08
            bottom = y
            for i, ln in enumerate(lines):
                bottom = self.text(ln, x, y + i * lh, style, color, gradient, alpha, scale, anchor, max_width)
            return bottom
        maxw = (max_width if max_width is not None else
                (SAFE["square"][2] - SAFE["square"][0] - 2 * SAFE["margin"]) * self.s)
        f0 = font(st["font"], max(8, int(round(st["size"] * scale * self.s))))
        w0 = sum(f0.getlength(ch) for ch in s) + st["tracking"] * scale * self.s * (len(s) - 1)
        if w0 > maxw:
            scale *= maxw / w0
        f = font(st["font"], max(8, int(round(st["size"] * scale * self.s))))
        trk = st["tracking"] * scale * self.s
        widths = [f.getlength(ch) for ch in s]
        total = sum(widths) + trk * (len(s) - 1)
        asc, desc = f.getmetrics()
        pad = int(60 * self.s * scale) + 4
        tw, th = int(total) + 2 * pad, asc + desc + 2 * pad
        mask = Image.new("L", (tw, th), 0)
        dm = ImageDraw.Draw(mask)
        cx = pad
        for ch, wch in zip(s, widths):
            dm.text((cx, pad), ch, font=f, fill=255)
            cx += wch + trk
        # anchor
        ox = {"l": 0, "m": total / 2, "r": total}[anchor[0]]
        oy = {"t": 0, "m": (asc + desc) / 2 * 0.92, "b": asc}[anchor[1]]
        X, Y = int(round(x - ox - pad)), int(round(y - oy - pad))
        m = np.asarray(mask, np.float32) / 255.0
        if gradient:
            cols = [C(n) for n in gradient]
            u = np.clip((np.arange(tw) - pad) / max(total, 1), 0, 1) * (len(cols) - 1)
            k = np.minimum(u.astype(int), len(cols) - 2)
            fr = (u - k)[:, None]
            row = np.array([cols[i] for i in k]) * (1 - fr) + np.array([cols[i + 1] for i in k]) * fr
            rgb = np.broadcast_to(row[None], (th, tw, 3))
        else:
            rgb = np.broadcast_to(C(color or st["color"]), (th, tw, 3))
        rgba = np.dstack([rgb, m * alpha])
        self._paste_overlay(rgba, X, Y)
        glow = st["glow"]
        if glow:
            g = np.asarray(mask.filter(ImageFilter.GaussianBlur(22 * self.s * scale)), np.float32) / 255.0
            self._add_glow(g[..., None] * C(glow) * 0.9 * alpha, X, Y)
        return Y + pad + asc + desc * 0.3   # bottom of this line, for stacking the next caption

    def _paste_overlay(self, rgba, X, Y):
        h, w = rgba.shape[:2]
        x0, y0, x1, y1 = max(X, 0), max(Y, 0), min(X + w, self.w), min(Y + h, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        src = rgba[y0 - Y:y1 - Y, x0 - X:x1 - X]
        im = Image.fromarray((np.clip(src, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
        self.overlay.alpha_composite(im, (x0, y0))

    def _add_glow(self, g, X, Y):
        h, w = g.shape[:2]
        x0, y0, x1, y1 = max(X, 0), max(Y, 0), min(X + w, self.w), min(Y + h, self.h)
        if x1 > x0 and y1 > y0:
            self.glow_layer[y0:y1, x0:x1] += g[y0 - Y:y1 - Y, x0 - X:x1 - X]

    def tag(self, episode: int, alpha=1.0):
        """The series tag ("SIM #04"), always in the same place and style."""
        x, y = TAG["xy"]
        s = TAG["format"].format(episode)
        self.dots([(x + 7 * self.s, y + 13 * self.s)], C(TAG["color"]), r=5, alpha=alpha)
        self.text(s, x + 22 * self.s, y + 13 * self.s, style=TAG["style"], color=TAG["color"], alpha=alpha, anchor="lm")

    # ---------------- finish ----------------
    def finish(self) -> np.ndarray:
        """Background + light + bloom, highlight roll-off, captions, dither -> HxWx3 uint8."""
        light = self.light
        bloom = np.zeros_like(light)
        for r, wgt in zip(GLOW["radii"], GLOW["weights"]):
            r = r * self.s
            f = 1
            while r / (f * 2) >= 1.25 and self.w % (f * 2) == 0 and self.h % (f * 2) == 0:
                f *= 2
            small = light.reshape(self.h // f, f, self.w // f, f, 3).mean((1, 3)) if f > 1 else light
            small = gaussian_blur(small, r / f)
            bloom += wgt * (_resize(small, self.w, self.h) if f > 1 else small)
        img = self._background() + light + GLOW["strength"] * bloom + self.glow_layer
        img = _rolloff(img)
        ov = np.asarray(self.overlay, np.float32) / 255.0
        a = ov[..., 3:4]
        img = img * (1 - a) + ov[..., :3] * a
        if GLOW["dither"]:
            rng = np.random.default_rng(self.index * 7919 + 17)
            img += (rng.random(img.shape, np.float32) - 0.5) / 255.0
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _rolloff(img: np.ndarray) -> np.ndarray:
    """Soft highlight roll-off. Above the knee values compress toward 1; hue_keep blends a
    hue-preserving version (overlaps stay coloured) with a per-channel one (overlaps whiten)."""
    k = GLOW["knee"]

    def f(v):
        over = np.maximum(v - k, 0, dtype=np.float32)
        return np.minimum(v, k) + (1 - k) * (1 - np.exp(-over / (1 - k)))

    m = img.max(-1, keepdims=True)
    hp = img * (f(m) / np.maximum(m, 1e-6))
    pc = f(img)
    h = GLOW["hue_keep"]
    return hp * h + pc * (1 - h)


# ============================================================================================
# Export
# ============================================================================================
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("simtheory_style: ffmpeg not found on PATH")
    return exe


def _video_args(out_path, audio_path, duration):
    e = EXPORT
    args = []
    if audio_path:
        args += ["-i", str(audio_path)]
    args += ["-map", "0:v:0"] + (["-map", "1:a:0"] if audio_path else [])
    args += ["-vf", f"scale=out_color_matrix={e['matrix']}:out_range={e['range']},format={e['pix_fmt']}",
             "-colorspace", e["matrix"], "-color_primaries", e["matrix"], "-color_trc", e["matrix"],
             "-color_range", e["range"],
             "-c:v", "libx264", "-preset", e["preset"], "-crf", str(e["crf"]), "-profile:v", e["profile"],
             "-movflags", "+faststart"]
    if audio_path:
        args += ["-af", f"loudnorm=I={e['lufs']}:TP={e['true_peak']}:LRA=11,aresample={e['audio_rate']}",
                 "-c:a", "aac", "-b:a", f"{e['aac_kbps']}k"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    return args + [str(out_path)]


class VideoWriter:
    """Pipe frames straight to ffmpeg with the series export settings.

        with VideoWriter("ep04.mp4", audio="score.wav") as vw:
            for n in range(frames):
                f = Frame(index=n); ...; vw.write(f.finish())
    """

    def __init__(self, out_path, audio=None, fps=None, w=W, h=H, duration=None):
        self.w, self.h = w, h
        fps = fps or EXPORT["fps"]
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        self.p = subprocess.Popen(cmd + _video_args(out_path, audio, duration), stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray):
        assert rgb.shape == (self.h, self.w, 3) and rgb.dtype == np.uint8, "frame must be HxWx3 uint8"
        self.p.stdin.write(rgb.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("simtheory_style: ffmpeg failed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_video(render_fn, n_frames, out_path, audio=None, workers=None, chunksize=4):
    """Render frames in parallel and stream them, in order, into an mp4 with the series export settings.
    render_fn(n) -> HxWx3 uint8 must be a top-level (picklable) function of the frame index."""
    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    first = render_fn(0)
    h, w = first.shape[:2]
    with VideoWriter(out_path, audio=audio, w=w, h=h, duration=n_frames / EXPORT["fps"]) as vw:
        vw.write(first)
        with mp.get_context("fork").Pool(workers) as pool:
            for k, fr in enumerate(pool.imap(render_fn, range(1, n_frames), chunksize=chunksize), 1):
                vw.write(fr)
                if k % (EXPORT["fps"] * 2) == 0:
                    print(f"  {k + 1}/{n_frames} frames", flush=True)


def encode_frames(pattern, out_path, audio=None, fps=None, duration=None):
    """Encode an existing PNG sequence (e.g. 'frames/f%05d.png') with the series export settings."""
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-framerate", str(fps or EXPORT["fps"]), "-i", str(pattern)]
    subprocess.run(cmd + _video_args(out_path, audio, duration), check=True)


def check_export(path) -> dict:
    """Return the colour tags of a video; raises if they are not the series standard."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,pix_fmt,color_space,color_primaries,color_transfer,color_range",
                          "-of", "default=nw=1", str(path)], capture_output=True, text=True, check=True).stdout
    tags = dict(line.split("=", 1) for line in out.strip().splitlines())
    want = {"color_space": EXPORT["matrix"], "color_primaries": EXPORT["matrix"],
            "color_transfer": EXPORT["matrix"], "color_range": EXPORT["range"]}
    bad = {k: tags.get(k) for k, v in want.items() if tags.get(k) != v}
    if bad:
        raise ValueError(f"simtheory_style: {path} has non-standard colour tags {bad}")
    return tags


# ============================================================================================
# Demo / test frame
# ============================================================================================
def _demo_curves(t, n=6, span=2.2, phase=0.0, amp=245):
    """Diverging trajectories for the test scene: same rule, growing offsets."""
    cx, cy = SAFE["subject"]
    out = []
    for k in range(n):
        ts = np.linspace(t - span, t, 120)
        off = 0.55 * k + phase
        x = cx + amp * np.sin(1.1 * ts + off) * np.cos(0.37 * ts + 0.4 * off)
        y = cy + amp * np.sin(1.7 * ts + 0.6 + 0.7 * off)
        out.append(np.stack([x, y], 1))
    return out


def demo_frame(t=3.2, index=0) -> np.ndarray:
    f = Frame(index=index)
    cx, cy = SAFE["subject"]
    # background crowd: faint ensemble gradient behind the role colours
    for b in range(6):
        f.lines(_demo_curves(t, n=8, span=0.9, phase=3.1 + b * 0.35 + 0.04 * np.arange(1), amp=300),
                ensemble_color(b / 5))
    curves = _demo_curves(t)
    roles = [role_color("reference"), role_color("main"), role_color("perturbed"),
             role_color("extra", 0), role_color("extra", 1), role_color("extra", 2)]
    for pts, col in zip(curves, roles):
        f.trail(pts, col)
        f.bob(*pts[-1], col)
    f.tag(4)
    bottom = f.text("CAN YOU\nPREDICT THIS?", 540, SAFE["text_y"], style="headline")
    f.text("Same rules. Tiny difference.", 540, bottom + 34, style="caption")
    # role legend in the HUD band
    names = [("reference", "white"), ("main", "cyan"), ("perturbed", "orange"),
             ("extra 1", "purple"), ("extra 2", "blue"), ("extra 3", "magenta")]
    y0 = SAFE["hud_band"][0] + 40
    for i, (lab, cname) in enumerate(names):
        x = 110 + (i % 3) * 300
        y = y0 + (i // 3) * 64
        f.dots([(x, y)], C(cname), r=9)
        f.text(lab, x + 24, y - 12, style="hud", color=cname, anchor="lm")
        f.text(COLORS[cname], x + 24, y + 14, style="hud", color="text_dim", anchor="lm", scale=0.85)
    return f.finish()


def _demo_clip_frame(n):
    return demo_frame(t=1.6 + n / FPS, index=n)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "demo":
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(demo_frame()).save(out / "style_test_frame.png")
        render_video(_demo_clip_frame, 3 * FPS, out / "style_test_clip.mp4")
        print("wrote", out / "style_test_frame.png", "and", out / "style_test_clip.mp4")
        print("colour tags:", check_export(out / "style_test_clip.mp4"))
    else:
        print(__doc__)
