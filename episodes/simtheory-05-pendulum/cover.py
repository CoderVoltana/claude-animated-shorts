"""Cover image: the glowing phase portrait (boundary at its brightest) with the hook text, 1080x1920 PNG."""
import os
import sys
from PIL import Image
import render

HERE = os.path.dirname(os.path.abspath(__file__))
fi = int(sys.argv[1]) if len(sys.argv) > 1 else 1578
r = render.Renderer()
r.cover = True
arr = r.frame(fi)
im = Image.fromarray(arr[..., :3]).resize((1080, 1920), Image.LANCZOS)
im.save(os.path.join(HERE, "out", "cover_1080x1920.png"), optimize=True)
print("cover from frame", fi, r.log["text"])
