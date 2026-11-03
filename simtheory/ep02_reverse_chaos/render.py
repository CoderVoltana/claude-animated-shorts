"""SimTheory #02 renderer, built on the series style kit (simtheory-style/simtheory_style.py).

Every colour, glow, size, font, text style, the SIM tag and the export come from the kit:
  GHOST  = role "reference" (white)      the recorded forward run played backwards (perfect rewind)
  CYAN   = role "main"                    reversed copy, float64 rounding only
  ORANGE = role "perturbed"               reversed copy, theta1 nudged by 1e-10 deg
  arrows = role "extra" 0 (purple)        velocity vectors at the reversal
No hex codes or font names appear in this file. Layout uses the kit's SAFE zones scaled to 1440x2560 (K = 4/3),
additionally kept out of the right 15 % / bottom 20 % for the YouTube Shorts UI.

    python render.py --still 16.6 out.png
    python render.py --video build/simtheory_02_reverse_chaos.mp4 --audio build/mix.wav
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "simtheory-style"))
import simtheory_style as S                                   # noqa: E402

import timeline as TL                                          # noqa: E402
from config import W, H, FPS, T_SIM, DUR                       # noqa: E402
from physics import DT, L, all_omegas                          # noqa: E402

EPISODE = 2
K = H / S.H                                                    # 4/3: kit layout is authored at 1080x1920

# ------------------------------------------------------------------ colours: kit roles only
REF = S.role_color("reference")
MAIN = S.role_color("main")
PERT = S.role_color("perturbed")
ARROW = S.role_color("extra", 0)
HUDC = S.C("hud")

# ------------------------------------------------------------------ layout (kit SAFE zones x K)
YT_RIGHT = W * 0.85                                            # Shorts UI column starts at x = 1224
TEXT_X = S.SAFE["subject"][0] * K                              # kit centre line (720)
TEXT_Y = S.SAFE["text_y"] * K                                  # 673
TEXT_MAXW = 2 * (YT_RIGHT - 16 - TEXT_X)                       # keeps captions left of the Shorts UI column
HUD_BAND = [v * K for v in S.SAFE["hud_band"]]                 # (1760, 2000): graph / readouts
PIV = (690.0, 1215.0)                                          # pivot: the full 2 m reach stays inside the
RPX = 240.0                                                    # square, above the HUD band, left of x = 1224
TAG_XY = (S.TAG["xy"][0] * K, S.TAG["xy"][1] * K)

# ------------------------------------------------------------------ data
D = np.load(os.path.join(HERE, "sim.npz"))
F, C, O = D["F"], D["C"], D["O"]
N = len(F) - 1
ANG_C, ANG_O = D["ang_c"], D["ang_o"]
FIT = TL.INFO["lyapunov_fit"]


def _bobs(Y):
    x1 = L * np.sin(Y[:, 0]); y1 = -L * np.cos(Y[:, 0])
    return np.stack([x1, y1, x1 + L * np.sin(Y[:, 1]), y1 - L * np.cos(Y[:, 1])], 1)


PF, PC, PO = _bobs(F), _bobs(C), _bobs(O)
ARC = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(PF[:, 2]), np.diff(PF[:, 3])))])


def sstep(a, b, t):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * u * (u * (6 * u - 15) + 10)


def zoom(t):
    """slow push-in during the rewind, easing back out at the reveal."""
    return 1.0 + 0.06 * sstep(11.0, 22.0, t) - 0.06 * sstep(23.6, 25.8, t)


class Cam:
    def __init__(self, t):
        self.z = zoom(t)
        self.s = RPX * self.z

    def xy(self, x, y):
        return PIV[0] + self.s * x, PIV[1] - self.s * y

    def pts(self, P):                                # (n, 2) world -> px
        return np.stack([PIV[0] + self.s * P[:, 0], PIV[1] - self.s * P[:, 1]], 1)


# ------------------------------------------------------------------ pendulum
def draw_copy(f, cam, P, color, a, ghost=False):
    if a <= 0.003:
        return
    p1 = cam.xy(P[0], P[1])
    p2 = cam.xy(P[2], P[3])
    k = 0.42 if ghost else 1.0                        # ghost: soft translucent white
    f.line([PIV, p1, p2], color, alpha=k * a)
    f.bob(*p1, color, alpha=k * a)
    f.bob(*p2, color, alpha=k * a)


def ghost_path(f, cam, k_now, a):
    """faint dashed path the ghost must still follow (its future = the forward record further back)."""
    if a <= 0.003:
        return
    i_now = N - k_now
    i_end = max(0, i_now - int(2.6 / DT))
    if i_now - i_end < 50:
        return
    idx = np.arange(i_now, i_end - 1, -25)
    P = cam.pts(PF[idx][:, 2:4])
    arc = cam.s * (ARC[-1] - ARC[idx])                 # dash pattern anchored to the track itself
    period = 30.0 * K * cam.z
    on = (arc % period) < period * 0.55
    bands = 5
    for b in range(bands):                             # fades with distance ahead
        lo, hi = len(P) * b // bands, len(P) * (b + 1) // bands
        dashes, cur = [], []
        for k in range(lo, min(hi + 1, len(P))):
            if on[k]:
                cur.append(P[k])
            else:
                if len(cur) >= 2:
                    dashes.append(np.array(cur))
                cur = []
        if len(cur) >= 2:
            dashes.append(np.array(cur))
        if dashes:
            f.lines(dashes, REF, width=1.5, alpha=0.32 * a * (1 - b / bands) ** 1.2)


_V = all_omegas(F[N:N + 1])[0]
_t1, _t2 = F[N, 0], F[N, 1]
V1 = np.array([L * _V[0] * math.cos(_t1), L * _V[0] * math.sin(_t1)])
V2 = V1 + np.array([L * _V[1] * math.cos(_t2), L * _V[1] * math.sin(_t2)])


def arrow(f, cam, P, v, sgn, a):
    """velocity arrow on a bob; sgn = signed scale (+1 forward, -1 reversed, passes 0 in the snap)."""
    if a <= 0.003 or abs(sgn) < 0.03:
        return
    sp = float(np.hypot(*v))
    Lpx = min(300.0, 120.0 + 22.0 * sp) * sgn * cam.z
    ux, uy = v[0] / sp, -v[1] / sp
    bx, by = cam.xy(*P)
    off = 22 * cam.z * np.sign(sgn)
    s0 = (bx + ux * off, by + uy * off)
    e = (bx + ux * Lpx, by + uy * Lpx)
    dx, dy = ux * np.sign(sgn), uy * np.sign(sgn)
    hd = 40 * cam.z * min(1.0, abs(sgn) * 1.6)
    nx, ny = -dy, dx
    h1 = (e[0] - dx * hd + nx * hd * 0.6, e[1] - dy * hd + ny * hd * 0.6)
    h2 = (e[0] - dx * hd - nx * hd * 0.6, e[1] - dy * hd - ny * hd * 0.6)
    w = S.SIZES["line"] * 1.7
    f.line([s0, e], ARROW, width=w, alpha=a)
    f.line([h1, e, h2], ARROW, width=w, alpha=a)


def ring(f, x, y, r, color, a):
    th = np.linspace(0, 2 * np.pi, 90)
    f.line(np.stack([x + r * np.cos(th), y + r * np.sin(th)], 1), color, alpha=a)


# ------------------------------------------------------------------ text
CAPTION_LINES = {   # the brief's exact captions, set in the kit's headline style (upper case + line breaks)
    "Can you reverse chaos?": "CAN YOU\nREVERSE CHAOS?",
    "Let it run wild…": "LET IT RUN WILD…",
    "Now reverse every velocity.": "NOW REVERSE\nEVERY VELOCITY.",
    "Physics says it should rewind perfectly.": "PHYSICS SAYS IT SHOULD\nREWIND PERFECTLY.",
    "Nudged by 0.0000000001°": "NUDGED BY\n0.0000000001°",
    "Just the computer’s rounding errors.": "JUST THE COMPUTER’S\nROUNDING ERRORS.",
    "Tiny errors grow exponentially.": "TINY ERRORS GROW\nEXPONENTIALLY.",
}


def captions(f, t):
    for (s, a0, a1, _acc) in TL.CAPTIONS:
        if not (a0 - 0.01 <= t <= a1 + 0.25):
            continue
        ain = sstep(a0, a0 + 0.2, t)
        aout = 1 - sstep(a1, a1 + 0.2, t)
        a = ain * aout
        if a <= 0.003:
            continue
        dy = 16 * (1 - ain) - 10 * (1 - aout)
        hook = s.startswith("Can you")
        f.text(CAPTION_LINES.get(s, s.upper()), TEXT_X, TEXT_Y + dy, style="headline", alpha=a,
               scale=1.0 if hook else 0.64, max_width=TEXT_MAXW)


def tag(f):
    """kit tag(), with the kit's position scaled to 2K (its sizes already scale with f.s)."""
    x, y = TAG_XY
    s = S.TAG["format"].format(EPISODE)
    f.dots([(x + 7 * f.s, y + 13 * f.s)], S.C(S.TAG["color"]), r=5)
    f.text(s, x + 22 * f.s, y + 13 * f.s, style=S.TAG["style"], color=S.TAG["color"], anchor="lm")


def hud(f, t):
    """direction + playback speed readout in the kit's HUD band (no sim-time counter). It gives the band to the
    graph during the reveal and is back, identical to frame 1, for the loop."""
    a = 1 - sstep(TL.T_GRAPH - 0.3, TL.T_GRAPH, t) + sstep(TL.T_GRAPH_OUT + 0.3, TL.T_GRAPH_OUT + 0.6, t)
    if a <= 0.003:
        return
    y = HUD_BAND[0] + 40
    rev = sstep(TL.T_FLIP - 0.04, TL.T_FLIP + 0.04, t) * (1 - sstep(29.25, 29.65, t))
    if 1 - rev > 0.003:
        f.text("\u25b6\u25b6 FORWARD", 100, y, style="hud", alpha=a * (1 - rev), anchor="lm")
    if rev > 0.003:
        f.text("\u25c0\u25c0 REVERSED", 100, y, style="hud", color=S.ROLES["extra"][0], alpha=a * rev, anchor="lm")
    f.text(f"PLAYBACK \u00d7{TL.speed(t):.1f}", YT_RIGHT - 44, y, style="hud", alpha=a, anchor="rm")


# ------------------------------------------------------------------ graph (kit HUD band)
GT = np.arange(0, N + 1, 200)
G_TAU = GT * DT
G_C = np.log10(np.maximum(ANG_C[GT], 1e-16))
G_O = np.log10(np.maximum(ANG_O[GT], 1e-16))
YMIN, YMAX = -16.5, 1.0


def graph(f, t):
    a = sstep(TL.T_GRAPH, TL.T_GRAPH + 0.3, t) * (1 - sstep(TL.T_GRAPH_OUT, TL.T_GRAPH_OUT + 0.3, t))
    if a <= 0.003:
        return
    b0, b1 = HUD_BAND
    px0, px1 = 240.0, 1180.0
    py0, py1 = b0 + 62, b1 - 50

    def X(tau):
        return px0 + (px1 - px0) * tau / T_SIM

    def Y(lg):
        return py1 - (py1 - py0) * (lg - YMIN) / (YMAX - YMIN)
    f.text("SEPARATION FROM THE PERFECT REWIND (rad)", 100, b0 + 18, style="hud", alpha=a, anchor="lm")
    f.line([(px0, py0 - 6), (px0, py1), (px1, py1)], HUDC, width=1.4, alpha=0.9 * a)
    for e, lab in ((-16, "1e-16"), (-8, "1e-8"), (0, "1")):
        f.line([(px0 + 4, Y(e)), (px1, Y(e))], HUDC, width=0.8, alpha=0.22 * a)
        f.text(lab, px0 - 16, Y(e), style="hud", alpha=a, anchor="rm")
    for s in (0, 5, 10, 15, 20):
        f.line([(X(s), py1), (X(s), py1 + 10)], HUDC, width=1.4, alpha=0.9 * a)
        f.text(str(s), X(s), py1 + 30, style="hud", alpha=a)
    f.text("s", px1, py1 + 30, style="hud", color="text_dim", alpha=a, anchor="rm")
    # data: drawn on as the graph opens, then following the live rewind
    reveal = sstep(TL.T_GRAPH + 0.05, TL.T_GRAPH + 1.0, t)
    n = int(np.searchsorted(G_TAU, TL.tau_rev(t) * reveal))
    for G, col in ((G_C, MAIN), (G_O, PERT)):
        if n >= 2:
            pts = np.stack([X(G_TAU[:n]), Y(np.clip(G[:n], YMIN, YMAX))], 1)
            f.line(pts, col, alpha=a)
            f.bob(*pts[-1], col, r=S.SIZES["bob"] * 0.55, alpha=a)
    # fitted exponentials (common window) = straight lines on the log axis, and their slope
    fa = a * sstep(TL.T_GRAPH + 1.0, TL.T_GRAPH + 1.5, t)
    if fa > 0.003:
        dashes = []
        for key, tb in (("common_cyan", 21.0), ("common_orange", 13.4)):
            fi = FIT[key]
            ts = np.linspace(0.0, tb, 60)
            lg = (fi["intercept"] + fi["slope"] * ts) / math.log(10)
            P = np.stack([X(ts), Y(lg)], 1)
            dashes += [P[i:i + 2] for i in range(0, len(P) - 1, 2)]
        f.lines(dashes, REF, width=1.2, alpha=0.55 * fa)
        f.text(f"λ ≈ {FIT['lambda_mean_common']:.2f} /s", px1 - 6, Y(-12.5), style="label", alpha=fa, anchor="rm")


# ------------------------------------------------------------------ frame
def frame_at(t, index=0):
    f = S.Frame(w=W, h=H, index=index, nebula_center=PIV)
    cam = Cam(t)
    ph = TL.phase(t)
    kr = min(N, int(round(TL.tau_rev(t) / DT)))
    kf = 0 if ph == "hook" else int(round(TL.sim_forward(t) / DT))
    reverse_on = t >= TL.T_REWIND
    if reverse_on:
        Pg, Pc, Po = PF[N - kr], PC[kr], PO[kr]
    else:
        Pg = Pc = Po = PF[kf]

    ignite = sstep(TL.T_RELEASE, TL.T_RELEASE + 0.25, t)              # copies switch on at the release
    out_ = 1 - sstep(*TL.T_FADE_COPIES, t)                           # and fade before the landing
    a_col = ignite * out_
    breathe = (1 - sstep(1.75, 2.0, t)) + sstep(28.6, 29.2, t)
    a_ghost = 1.0 + 0.12 * breathe * math.cos(2 * math.pi * t / 1.5)  # 20 periods per loop: seamless

    f.pivot(*PIV)
    if ph in ("forward", "freeze") and t >= TL.T_RELEASE:
        fade = 1 - sstep(9.3, 10.8, t)
        idx = np.arange(max(0, kf - int(2.2 / DT)), kf + 1, 60)
        if len(idx) > 2:
            P = cam.pts(PF[idx][:, 2:4])
            f.trail(P, MAIN, alpha=fade * a_col)
            f.trail(P, PERT, alpha=fade * a_col)
    if reverse_on:
        ghost_path(f, cam, kr, sstep(11.0, 11.6, t))
        idx = np.arange(max(0, kr - int(1.3 / DT)), kr + 1, 40)
        if len(idx) > 2:
            f.trail(cam.pts(PC[idx][:, 2:4]), MAIN, alpha=out_)
            f.trail(cam.pts(PO[idx][:, 2:4]), PERT, alpha=out_)

    draw_copy(f, cam, Pg, REF, a_ghost, ghost=True)
    draw_copy(f, cam, Pc, MAIN, a_col)
    draw_copy(f, cam, Po, PERT, a_col)

    if 9.0 <= t <= 11.6:                                              # velocity arrows and the snap
        a_ar = sstep(9.15, 9.45, t) * (1 - sstep(11.0, 11.45, t))
        u = (t - TL.T_FLIP) / 0.16
        if u <= 0:
            sg = 1.0
        elif u >= 1:
            dt_ = t - TL.T_FLIP - 0.16
            sg = -1.0 - 0.18 * math.exp(-dt_ * 14) * math.cos(dt_ * 40)
        else:
            sg = math.cos(math.pi * u * u)
        Pa = PF[N]
        arrow(f, cam, Pa[0:2], V1, sg, a_ar)
        arrow(f, cam, Pa[2:4], V2, sg, a_ar)
        if t >= TL.T_FLIP + 0.08:
            fl = math.exp(-(t - TL.T_FLIP - 0.08) * 9)
            if fl > 0.01:
                for (bx, by) in (cam.xy(*Pa[0:2]), cam.xy(*Pa[2:4])):
                    ring(f, bx, by, 30 + 130 * (1 - fl), ARROW, 0.9 * fl)

    graph(f, t)
    captions(f, t)
    hud(f, t)
    tag(f)
    return f.finish()


def render_frame(n):
    return frame_at(n / FPS, index=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs=2)
    ap.add_argument("--video")
    ap.add_argument("--audio")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--crf", type=int, default=6)          # kit master (near-lossless)
    ap.add_argument("--final_crf", type=int, default=9)    # delivery, ~45 Mb/s
    args = ap.parse_args()
    if args.still:
        from PIL import Image
        t = float(args.still[0])
        Image.fromarray(frame_at(t, index=int(round(t * FPS)))).save(args.still[1])
        return
    # Kit export (BT.709 matrix + tags, yuv420p, High profile) with two documented exceptions from the brief:
    #  * quality: high bitrate (the kit's CRF 15 gives ~5 Mb/s here): a CRF 6 kit master, then a CRF 9 delivery file;
    #  * audio: mix.wav is already mastered to -14 LUFS / -1.7 dBTP with a seamless loop, so it is muxed as is
    #    (the kit's single-pass loudnorm re-compresses it to about -12 LUFS).
    import subprocess
    S.EXPORT["crf"] = args.crf
    silent = args.video.replace(".mp4", "_video.mp4")
    S.render_video(render_frame, int(round(DUR * FPS)), silent, audio=None, workers=args.workers)
    # final: re-encode the near-lossless kit master (YUV -> YUV, no colour conversion, same BT.709 tags) to ~45 Mb/s
    # and mux the mastered audio
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", silent]
    if args.audio:
        cmd += ["-i", args.audio, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", f"{S.EXPORT['aac_kbps']}k",
                "-ar", str(S.EXPORT["audio_rate"])]
    cmd += ["-c:v", "libx264", "-preset", S.EXPORT["preset"], "-profile:v", S.EXPORT["profile"], "-crf", str(args.final_crf),
            "-pix_fmt", S.EXPORT["pix_fmt"], "-colorspace", S.EXPORT["matrix"], "-color_primaries", S.EXPORT["matrix"],
            "-color_trc", S.EXPORT["matrix"], "-color_range", S.EXPORT["range"], "-t", f"{DUR:.3f}",
            "-movflags", "+faststart", args.video]
    subprocess.run(cmd, check=True)
    print(S.check_export(args.video))


if __name__ == "__main__":
    main()
