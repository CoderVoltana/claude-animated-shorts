"""QA helper: contact sheet of stills with the safe-area box drawn on top."""
import sys
from PIL import Image, ImageDraw
paths = sys.argv[2:]
ims = []
for p in paths:
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 384, 1152, 1664], outline=(255, 0, 0), width=4)
    ims.append(im.resize((480, 853)))
cols = 4
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 480, rows * 853), (40, 40, 40))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * 480, (i // cols) * 853))
sheet.save(sys.argv[1])
