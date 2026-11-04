"""Contact sheet of stills (for visual inspection). Usage: python sheet.py out.jpg f1.png f2.png ... [--w 480]"""
import sys
from PIL import Image, ImageDraw
args = sys.argv[1:]
w = 480
if "--w" in args:
    i = args.index("--w"); w = int(args[i + 1]); del args[i:i + 2]
out, files = args[0], args[1:]
h = w * 16 // 9
cols = min(4, len(files))
rows = (len(files) + cols - 1) // cols
S = Image.new("RGB", (cols * (w + 8), rows * (h + 8)), (40, 40, 40))
for k, f in enumerate(files):
    im = Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    # safe-area box (scaled) for inspection only
    s = w / 1440
    d.rectangle([86 * s, 384 * s, 1152 * s, 1664 * s], outline=(255, 0, 0))
    d.text((6, h - 16), f.split("/")[-1], fill=(255, 255, 0))
    S.paste(im, ((k % cols) * (w + 8), (k // cols) * (h + 8)))
S.save(out, quality=90)
