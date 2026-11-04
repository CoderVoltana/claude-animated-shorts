"""contact sheet of stills with the safe area drawn (for inspection only)"""
import sys, cv2, numpy as np
sys.path.insert(0, __file__.rsplit('/',2)[0]); import timeline as T
paths = sys.argv[2:]; out = sys.argv[1]
ims = []
for p in paths:
    im = cv2.imread(p)
    x0,y0,x1,y1 = T.SAFE
    cv2.rectangle(im, (x0,y0), (x1,y1), (0,0,255), 3)
    im = cv2.resize(im, (540, 960), interpolation=cv2.INTER_AREA)
    cv2.putText(im, p.rsplit('_',1)[-1][:-4], (8, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    ims.append(im)
while len(ims) % 4: ims.append(np.zeros_like(ims[0]))
rows = [np.hstack(ims[i:i+4]) for i in range(0, len(ims), 4)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
