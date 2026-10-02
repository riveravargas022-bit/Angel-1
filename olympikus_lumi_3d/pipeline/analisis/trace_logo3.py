import numpy as np
from PIL import Image
from scipy import ndimage as ndi
im = np.asarray(Image.open('ref/652LUMILILA_1.jpg').convert('RGB')).astype(float)
x0,y0,x1,y1 = 1700,1400,2700,1800
c = im[y0:y1, x0:x1]
R,G,B = c[...,0],c[...,1],c[...,2]
lum = (R+G+B)/3
sat = B - (R+G)/2
mint = (G - R > 18) & (G > B - 8)
stroke = ((lum < 168) & (sat > 14)) | mint
stroke = ndi.binary_closing(stroke, structure=np.ones((3,3)), iterations=2)
lab, n = ndi.label(stroke)
sizes = ndi.sum(stroke, lab, range(1, n+1))
keep = np.zeros_like(stroke)
for k, s in enumerate(sizes):
    if s > 4000: keep |= (lab == k+1)
holes = ndi.binary_fill_holes(keep) & ~keep
hl, hn = ndi.label(holes); hs = ndi.sum(holes, hl, range(1, hn+1))
for k, s in enumerate(hs):
    if s < 600: keep |= (hl == k+1)
keep = ndi.binary_opening(keep, iterations=2)
print('px', keep.sum(), 'n comps', ndi.label(keep)[1])
vis = (c*0.5+127).astype(np.uint8); vis[keep] = (90, 60, 200); vis[mint & keep] = (0, 200, 120)
Image.fromarray(vis).save('out/logo_trace3.png')
np.save('tex/logo_keep.npy', keep); np.save('tex/logo_mint.npy', mint & keep)
