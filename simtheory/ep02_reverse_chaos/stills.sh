#!/bin/bash
# usage: stills.sh outdir t1 t2 ...   -> renders stills in parallel and a contact sheet outdir/sheet.png
out=$1; shift; mkdir -p $out; rm -f $out/f_*.png
for t in "$@"; do python3 render.py --still $t $out/f_$t.png 2>/dev/null & done; wait
python3 - "$out" <<'PY'
import sys, glob
from PIL import Image, ImageDraw
o=sys.argv[1]
fs=sorted(glob.glob(o+'/f_*.png'), key=lambda p: float(p.split('_')[-1][:-4]))
n=len(fs); cols=min(4,n); rows=(n+cols-1)//cols
sheet=Image.new('RGB',(370*cols,650*rows),(40,40,40))
for i,f in enumerate(fs):
    im=Image.open(f).convert('RGB').resize((360,640)); d=ImageDraw.Draw(im)
    d.line([(306,0),(306,640)],fill=(90,0,0)); d.line([(0,512),(360,512)],fill=(90,0,0))
    d.text((5,620),f.split('_')[-1][:-4],fill=(255,255,0)); sheet.paste(im,((i%cols)*370,(i//cols)*650))
sheet.save(o+'/sheet.png')
PY
