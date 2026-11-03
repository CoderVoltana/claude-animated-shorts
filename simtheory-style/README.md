# SimTheory style kit

`simtheory_style.py` is one file that draws, glows, captions and encodes every SimTheory episode the same way. Colours can't drift between chats, because each chat uses the same rendering code instead of re-inventing it.

## Using it in a new chat
Attach `simtheory_style.py` **and the `fonts/` folder**, then add this line to the prompt:

> Use the attached `simtheory_style.py` for all drawing, glow, text and encoding. Use its colour roles, and don't define colours, fonts, glow or export settings yourself.

Requirements: Python 3.9+, numpy, Pillow, and ffmpeg.

To check that it works: `python simtheory_style.py demo out/` writes a test frame and a 3 s test clip, and confirms the clip's colour tags.

## Colour roles (fixed for the whole series)
| Role | Colour | Hex |
|---|---|---|
| Reference / true result | White | `#F4F7FF` |
| Main / ideal case | Cyan | `#22E5FF` |
| Perturbed / "different" case | Orange | `#FF8A1E` |
| Extra case 1 | Purple | `#A848FF` |
| Extra case 2 | Blue | `#3A7BFF` |
| Extra case 3 | Magenta | `#FF3FA4` |
| Background crowds | Cyan → blue → purple gradient, 16% opacity | |

In code, use `role_color("main")`, `role_color("extra", 1)`, `ensemble_color(u)` and `C("cyan")`. Never type hex values.

## What it fixes
- **Base colours:** exact hex codes live only in `COLORS`.
- **Glow:** one bloom recipe (`GLOW`: 5 / 16 / 44 px radii, fixed weights). Highlights roll off softly and keep their hue, so overlapping lines brighten without bleaching to white.
- **Encoding:** RGB → YUV uses the BT.709 matrix, and the file is tagged BT.709 / limited range. On solid swatches the kit round-trips every role colour within 2/255. ffmpeg's defaults shift cyan by −19 and magenta by +22.
- **Sizes, type and layout:**
  - Line, trail and bob sizes are fixed (`SIZES`).
  - Fonts are bundled (Inter Display Black, Inter SemiBold and JetBrains Mono). If a font is missing, the kit stops instead of silently switching fonts.
  - Caption styles are fixed (`TEXT`).
  - The `SIM #0X` tag is drawn the same way every time (`TAG`).
  - Safe zones are defined in `SAFE`: everything readable goes inside the centred 1080×1080 square, which survives YouTube, Instagram's square, 4:5 and 3:4 crops, and the Reels interface.

## Minimal episode
```python
import simtheory_style as S

def frame(n):
    f = S.Frame(index=n)
    f.trail(points_main, S.role_color("main"))           # pts oldest -> newest, in px
    f.trail(points_pert, S.role_color("perturbed"))
    f.bob(*points_main[-1], S.role_color("main"))
    f.tag(4)                                              # SIM #04
    y = f.text("CAN YOU\nPREDICT THIS?", 540, S.SAFE["text_y"], style="headline")
    f.text("Same rules. Tiny difference.", 540, y + 34, style="caption")
    return f.finish()

S.render_video(frame, 24 * S.FPS, "ep04.mp4", audio="score.wav")   # parallel render + series export
S.check_export("ep04.mp4")                                            # raises if the colour tags are wrong
```
Frames rendered elsewhere (for example in a browser) can be encoded the same way with `S.encode_frames("frames/f%05d.png", "ep.mp4", audio="score.wav")`.

## Files
- `simtheory_style.py`: the kit (version in `VERSION`).
- `fonts/`: Inter (SIL Open Font License) and JetBrains Mono (Apache-2.0), with licence files.
- `examples/style_test_frame.png` and `examples/style_test_clip.mp4`: reference output to compare new renders against.
