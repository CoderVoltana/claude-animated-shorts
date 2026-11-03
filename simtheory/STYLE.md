# SimTheory: series style (keep identical across episodes)

| Element | Value |
|---|---|
| Canvas | 1440×2560 (9:16), 60 fps, pure black `#000000` background |
| Safe zone | Nothing important in the bottom 20 % (y > 2048) or the right 15 % (x > 1224). Content is centred on x = 612. |
| Episode tag | `SIM #NN`, top-left (x 64, baseline 196), Inter Display **Light** 50 px, tracking +6 px, cyan `#00BEFF` |
| Captions | Inter Display **Bold** 92 px, white, centred on x = 612, block centre y = 640, wrap at 1060 px, line height 106 px. Soft black shadow (blur 14). 0.2 s smootherstep fades with a 16 px rise. An optional 140×10 px accent bar in the colour of the subject. |
| HUD | Top-right, right edge x = 1184. "SIM TIME" label Inter Medium 30 px (+4 tracking); counter Inter Medium 50 px with fixed-width digits; direction Inter SemiBold 40 px with drawn ▶▶ / ◀◀; playback speed `×N.N` Inter SemiBold 40 px |
| Palette | cyan `#00BEFF` (core `#8CE4FF`), orange `#FF6200` (core `#FFB270`), ghost white at 9–55 % alpha, neon purple `#BE46FF` (core `#E6B9FF`). Cyan + orange add up to about white, so aligned copies read as one white pendulum. |
| Rendering | Additive (`plus`) blending for every emissive element; bloom = ¼-res copy blurred at σ 14 px and σ 44 px (full-res equivalents), added at 0.85 and 0.75 |
| Line weights (at 1440 wide) | arms 13 px core in a 30 px halo, bobs r = 27 px, trails 5.5 to 7 px, arrows 13 px shaft in a 34 px halo, graph curves 5.5 px in a 16 px halo, axes 2.5 px |
| Camera | Slow zoom about the pivot only (no cuts, no shake) |
| Audio | Periodic drone (every partial a whole number of cycles per loop), −14 LUFS integrated, true peak ≤ −1 dBTP, a sample-continuous loop point |
| Fonts | Inter / Inter Display (SIL OFL), from `/usr/share/fonts/opentype/inter/` |
