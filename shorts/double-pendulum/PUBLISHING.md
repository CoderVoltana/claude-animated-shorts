# Publishing: Double Pendulum Chaos

Video: `out/double_pendulum_chaos.mp4` (24 s, 1080x1920, 60 fps, loops seamlessly)

## Title
- **A Billionth of a Degree Changed Everything 🤯** (recommended)
- Can You Predict a Double Pendulum?
- 500 Pendulums. Same Start. Total Chaos.

## Description
500 double pendulums, each started just 0.000000001° apart. For 14 seconds they move as one.
Then chaos takes over. Same physics, no randomness, completely different futures.
Simulated with the exact equations of motion (RK4, energy conserved to 10⁻⁹ J).

#physics #chaostheory #doublependulum #butterflyeffect #science #simulation #math #shorts

## Hashtags
#physics #chaostheory #doublependulum #butterflyeffect #science #simulation #math #shorts
(YouTube shows the first three above the title.)

## Thumbnails (`out/thumbnails/`)
| File | Video frame | Use |
|---|---|---|
| `thumb_C_number.png` | 16.9 s + text overlay | Recommended if custom thumbnail upload is available. Pairs with the recommended title. |
| `thumb_D_chaos_clean.png` | 19.6 s, clean render (`CLEAN=1`) | Only the CHAOS title and the pendulums: no sublines, graph or legend. Needs custom upload, since this exact frame isn't in the video. |
| `thumb_A_chaos.png` | 17.5 s | Best pick in the in-app frame selector: CHAOS title plus the burst. |
| `thumb_B_tracked.png` | 19.6 s | Alternative frame: the 5 tracked pendulums and their starting angles. |

## Instagram
`out/thumbnails/instagram_post_4x5.png`: 1080x1350 (4:5 portrait, the largest feed format).
It's a crop of the clean 19.6 s render. The profile grid's 3:4 crop trims about 34 px per side, and the title stays clear of it.
`out/thumbnails/instagram_reel_cover.png`: 1080x1920 Reel cover, rendered with the square-safe layout (`IG=1 CLEAN=1`). Content runs from y 466 to 1347, inside the centred 1:1 crop.

### Instagram Reel video
Upload `out/double_pendulum_chaos_instagram.mp4`, not the YouTube file. It's the same 24 s short, but every element stays inside the centred 1080x1080 square (y 420–1500). It survives Instagram's square crop, the 4:5 feed, the 3:4 grid and full-screen 9:16 with its UI overlays.
Render it with `IG=1 node render.mjs frames <dir> 4`.
