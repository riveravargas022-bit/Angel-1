import numpy as np, json
import shoe_geom as g
fg = np.load('ref/fg_5.npy')
info = json.load(open('tex/outsole_info.json'))
sole = fg.copy(); sole[3925:, :] = False
rows = np.nonzero(sole.any(1))[0]
rt, rh = rows.min(), rows.max()            # toe (top), heel (bottom)
# row = r0 + r1*u  (u=1 toe at rt, u=0 heel at rh)
r0, r1 = rh, rt - rh
U = np.linspace(0.06, 0.94, 45)
A, b = [], []
for u in U:
    r = int(round(r0 + r1 * u))
    c = np.nonzero(sole[r])[0]
    wl, wm = g.footprint(np.array([u]))
    # left image edge = lateral (y=-wl*L), right = medial (y=+wm*L)
    A.append([1, u, -wl[0] * g.L]); b.append(c.min())
    A.append([1, u, wm[0] * g.L]); b.append(c.max())
A = np.array(A); b = np.array(b, float)
(c0, c1, k), *_ = np.linalg.lstsq(A, b, rcond=None)
res = A @ np.array([c0, c1, k]) - b
print('fit c0=%.1f c1=%.1f k=%.1f px/m  (length scale %.1f px/m)  rms=%.1f px' % (c0, c1, k, abs(r1) / g.L, np.sqrt((res ** 2).mean())))
x0, x1, y0, y1 = info['x0'], info['x1'], info['y0'], info['y1']
# V = 1 - (row - y0)/(y1-y0), row = r0 + r1*u, u = x/L + 0.5
H = (y1 - y0); W = (x1 - x0)
av = -r1 / H / g.L
bv = 1 - (r0 + r1 * 0.5 - y0) / H
# U = (c0 + c1*u + k*y - x0)/W
aux = c1 / W / g.L
auy = k / W
bu = (c0 + c1 * 0.5 - x0) / W
json.dump(dict(av=av, bv=bv, aux=aux, auy=auy, bu=bu), open('tex/outsole_fit.json', 'w'))
print(json.load(open('tex/outsole_fit.json')))
