# SimTheory #03: What if gravity was 10× stronger?

**Video:** `out/SimTheory_03_10x_Gravity.mp4` (1440×2560, 60 fps, H.264 High, BT.709, AAC 256 kb/s, 30.0 s, loops seamlessly)

## Upload text

**Title** (49 characters)
```
What If Gravity Was 10× Stronger? | SimTheory #03
```

**Description**
```
10× stronger gravity should swing a pendulum 10× faster… but it's only 3.16× faster, because gravity hides under a square root: T = 2π√(L/g).
Real physics: full nonlinear pendulum simulation, no tricks.
```

**Hashtags**
```
#physics #science #simulation #gravity #shorts
```

**Pinned comment**
```
🧪 The next episode is up to you: what should we simulate next? Drop your "what if…" below, and the most-liked idea gets built and simulated for real 👇
```

## Measured numbers (from `sim.py`, also in `physics_log.txt`)

| | Earth (g = 9.81) | 10× gravity (g = 98.1) |
|---|---|---|
| Period, simulated (RK4, dt = 0.5 ms, θ₀ = 20°) | **2.021451 s** | **0.639239 s** |
| Exact elliptic-integral period (reference check) | 2.021451 s | 0.639239 s |
| Small-angle period (not used) | 2.006067 s | 0.634374 s |
| Full swings at the sync | 6 (at 12.128708 s) | 19 (at 12.145541 s) |

- **Ratio T_Earth / T_10×: 3.162278** (√10 = 3.162278). Spec 3.162 ± 0.005: **PASS**
- **Sync:** both are back at the release point at **12.145541 s** of sim time, which is **14.146 s** in the video (the freeze is frame 849). The two returns are 16.8 ms apart, a 0.139% mismatch. Spec < 0.2%: **PASS**
- Energy drift over the run: 8×10⁻¹⁶ (Earth), 3×10⁻¹³ (10×). No damping, as specified.
- 100× ghost: g = 981 m/s², period 0.2021 s, also simulated with RK4.

Note: the brief estimated the sync at "~12.04 s". That is the small-angle value, 6 × 2.006 s. At 20° the real nonlinear pendulum runs 0.77% slower, so the true sync is at 12.146 s. The video uses the simulated value.

## Timeline as built (30.0 s)

| Time | Beat |
|---|---|
| 0.00–2.00 | Hook. Both pendulums held at 20°, counters at 0, caption "10× stronger gravity…". Frame 1 is fully lit, with no fade from black. |
| 2.00 | Release, exactly at frame 120. Caption "…10× faster swing?" |
| 4.05–14.15 | Race in real time. Caption "Count the swings." Each counter pops and its bob shows a ring on every full swing. |
| 14.146 | Freeze at the sync point with counters at 6 and 19. One chord, then a brief silence. |
| 14.33–21.95 | Reveal. Captions "Only 3.16× faster." and "Gravity hides under a square root." KaTeX formula from 15.55; the √ glows and pulses from 16.5. Slow push-in (to 1.09×), then easing back out. |
| 21.85–27.15 | Punchline. Translucent purple 100× ghost (real sim, motion-blurred) flickers in with a buzzing whir. Caption "For 10× faster, you'd need 100× gravity." |
| 27.45–30.00 | Loop. Counters reset to 0 and the hook text returns at 27.55, with the pendulums at 20°. The last frame flows into frame 1. |

The brief's timeline added up to 25 s, but the requested length was 30–40 s. The race is fixed by the physics, so the extra 5 s went to the reveal and the punchline, which gives the formula and the 100× ghost time to read.

## Series style
The picture is drawn entirely with the SimTheory style kit (`simtheory-style/simtheory_style.py`, v1.0.0):
- **Colour roles:** Earth = main (cyan), 10× gravity = perturbed (orange), 100× ghost = extra 1 (purple), formula = reference (white), divider = HUD blue.
- **Kit look:** nebula backdrop, `GLOW` bloom and highlight roll-off, `SIZES` for rods, bobs, trails and sparks.
- **Text styles:** captions use `headline` (Inter Display Black with cyan glow), labels use `label` (JetBrains Mono), counters use `number`.
- **Export:** the kit's export (CRF 15, BT.709 limited range, tags confirmed by `check_export`).

Episode-specific choices:
- The `SIM #03` tag is not shown.
- The layout follows the brief's safe area (clear of the top 15%, bottom 35% and right 20%), not the kit's centred square, which reaches into the bottom 35%.
- The audio is mastered to the kit's targets (−14 LUFS, ≤ −1.5 dBTP) and muxed directly. It is not run through ffmpeg loudnorm, so the loop seam stays sample-continuous.

## Rebuild
```
./build.sh
```
Needs Python 3 with numpy, scipy and Pillow, Node with Playwright/Chromium (KaTeX is installed from npm), and ffmpeg. Fonts come bundled with the style kit in `simtheory-style/fonts/`.
