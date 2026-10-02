import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as ndi
S = json.load(open('spec.json'))
res = {}
for i, name in ((1,'lat'), (2,'med')):
    im = cv2.imread(f'ref/652LUMILILA_{i}.jpg')[:, :, ::-1].astype(np.float32)
    side = S['side%d' % i]; x0, x1, gr = side['x0'], side['x1'], side['ground']; Lp = x1 - x0
    lum = im.mean(2)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.5)
    tex = cv2.GaussianBlur(np.abs(hp), (0, 0), 4)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    smooth = (tex < 1.6) & (lum > 120) & (lum < 235)
    H, W = lum.shape
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - x0) / Lp
    if i == 2: u = 1 - u
    h = (gr - yy) / Lp
    zone = (u > 0.33) & (u < 0.80) & (h > 0.20) & (h < 0.42)
    m = smooth & zone
    m = ndi.binary_opening(m, iterations=3)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n+1))
    big = np.argsort(-sizes)[:3] + 1
    m = np.isin(lab, big)
    m = ndi.binary_closing(m, iterations=6)
    m = ndi.binary_fill_holes(m)
    out = []
    for uu in np.arange(0.34, 0.80, 0.01):
        col = int(round(x0 + uu*Lp)) if i == 1 else int(round(x1 - uu*Lp))
        rows = np.nonzero(m[:, col])[0]
        if len(rows) == 0: continue
        out.append((round(float(uu), 3), round(float((gr - rows.max())/Lp), 4), round(float((gr - rows.min())/Lp), 4)))
    res[name] = out
    vis = (im*0.5 + 127).astype(np.uint8); vis[m] = (vis[m]*0.4 + np.array([255, 0, 120])*0.6).astype(np.uint8)
    Y0, Y1 = int(gr - 0.47*Lp), int(gr - 0.18*Lp)
    X0, X1 = (int(x0+0.30*Lp), int(x0+0.82*Lp)) if i == 1 else (int(x1-0.82*Lp), int(x1-0.30*Lp))
    Image.fromarray(vis[Y0:Y1, X0:X1]).resize(((X1-X0)//3, (Y1-Y0)//3)).save(f'out/es_seg_{name}.png')
json.dump(res, open('ref/eyestay_bounds.json', 'w'))
for k, v in res.items():
    print(k, v)
