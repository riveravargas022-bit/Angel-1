"""Vectorize the Olympikus mark from the mint-inlay skeleton traced on the lateral photo."""
import numpy as np, json
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw
sk = np.load('tex/mint_skel.npy')
H, W = sk.shape
pts = set(map(tuple, np.argwhere(sk)))
def nbrs(p):
    y, x = p
    return [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and (y+dy, x+dx) in pts]
deg = {p: len(nbrs(p)) for p in pts}
nodes = {p for p in pts if deg[p] != 2}
# cluster junction pixels
nodes_l = list(nodes)
visited_edges = set()
paths = []
for n0 in nodes_l:
    for nb in nbrs(n0):
        if (n0, nb) in visited_edges: continue
        path = [n0, nb]; prev, cur = n0, nb
        visited_edges.add((n0, nb))
        while cur not in nodes:
            nxt = [q for q in nbrs(cur) if q != prev and q not in path[-3:]]
            if not nxt: break
            # prefer 4-connected continuation
            nxt.sort(key=lambda q: abs(q[0]-cur[0]) + abs(q[1]-cur[1]))
            prev, cur = cur, nxt[0]
            path.append(cur)
        visited_edges.add((cur, prev))
        if len(path) > 8:
            paths.append(path)
# dedupe paths (same endpoints reversed)
uniq = []
seen = set()
for p in paths:
    key = tuple(sorted([p[0], p[-1]])) + (len(p)//5,)
    if key in seen: continue
    seen.add(key); uniq.append(p)
print('paths', len(uniq), [len(p) for p in uniq])
# merge junction clusters: snap path endpoints that are within 4px to a common centroid
ends = np.array([p[0] for p in uniq] + [p[-1] for p in uniq], float)
lab = -np.ones(len(ends), int); c = 0
for i in range(len(ends)):
    if lab[i] >= 0: continue
    d = np.linalg.norm(ends - ends[i], axis=1); m = (d < 7) & (lab < 0); lab[m] = c; c += 1
cent = np.array([ends[lab == k].mean(0) for k in range(c)])
out = []
for i, p in enumerate(uniq):
    a = np.array(p, float)
    a[0] = cent[lab[i]]; a[-1] = cent[lab[i + len(uniq)]]
    # resample uniformly
    seg = np.linalg.norm(np.diff(a, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / 2), 6)
    t = np.linspace(0, s[-1], n)
    r = np.stack([np.interp(t, s, a[:, k]) for k in range(2)], 1)
    if n > 9:
        sm = savgol_filter(r, min(15, n - (1 - n % 2)), 3, axis=0)
        sm[0], sm[-1] = r[0], r[-1]
        # blend ends to keep junction positions
        r = sm
    out.append(r[:, ::-1].tolist())   # (x, y)
endpoints = [e[::-1].tolist() for e in cent]
deg_c = [int((lab == k).sum()) for k in range(c)]
json.dump({'paths': out, 'nodes': endpoints, 'node_degree': deg_c, 'crop_origin': [1700, 1400]}, open('tex/logo_paths.json', 'w'))
print('nodes', [(np.round(e,1).tolist(), d) for e, d in zip(endpoints, deg_c)])
