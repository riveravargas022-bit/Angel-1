"""Secondary components of the shoe (pure numpy): upper mesh, collar padding, tongue,
eyestay overlay, eyelets, laces, heel pull tab, insole, side badge."""
import json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator, CubicSpline
import shoe_geom as g

L = g.L
HERE = g.HERE
MM = 1e-3


def norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def resample_poly(P, n):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])], 1), s[-1]


def smooth_poly(P, sigma, keep_ends=True):
    Q = gaussian_filter1d(P, sigma, axis=0, mode='nearest')
    if keep_ends:
        Q[0], Q[-1] = P[0], P[-1]
    return Q


def tube_faces(n_along, n_ring, closed_ring=True, cap=False):
    F = []
    for i in range(n_along - 1):
        for j in range(n_ring if closed_ring else n_ring - 1):
            j2 = (j + 1) % n_ring
            F.append((i * n_ring + j, i * n_ring + j2, (i + 1) * n_ring + j2, (i + 1) * n_ring + j))
    return F


# ----------------------------------------------------------------------------
# Upper
# ----------------------------------------------------------------------------
class Upper:
    def __init__(self):
        r = g.upper_surface_v2()
        grid = r['grid']
        self.r = r
        self.grid = grid
        self.u, self.p = r['u'], r['p']
        n, nt, _ = grid.shape
        self.rside = np.array([ri[0] for ri in r['rinfo']])
        self.ru = np.array([ri[1] for ri in r['rinfo']])
        self.rpsi = np.array([ri[2] if ri[2] is not None else np.nan for ri in r['rinfo']], float)
        F = []
        for i in range(n - 1):
            for j in range(nt - 1):
                F.append((i * nt + j, i * nt + j + 1, (i + 1) * nt + j + 1, (i + 1) * nt + j))
        Vw, Fw, inv = g.weld(grid.reshape(-1, 3), F, 1e-6)
        self.V, self.F, self.inv = Vw, Fw, inv
        uu = np.repeat(self.ru, nt)
        self.vu = np.zeros(len(Vw)); self.vu[inv] = uu
        tt = np.tile(np.linspace(0, 1, nt), n)
        self.vtau = np.zeros(len(Vw)); self.vtau[inv] = tt
        self.N = g.vertex_normals_poly(Vw, Fw)
        # outward check: compare with direction from a interior axis point
        yt = np.interp(self.vu, self.u, self.p['yt'])
        c = np.stack([Vw[:, 0], yt, np.full(len(Vw), 0.075)], 1)
        out = Vw - c
        out[:, 2] *= 0.3
        self.flipped = ((self.N * out).sum(1) < 0).mean() > 0.5
        if self.flipped:
            self.N = -self.N
        self.tree = cKDTree(Vw)

    def project(self, Q, offset=0.0):
        d, idx = self.tree.query(Q, k=4)
        w = 1.0 / np.maximum(d, 1e-6)
        w /= w.sum(1, keepdims=True)
        P0 = (self.V[idx] * w[..., None]).sum(1)
        N0 = norm((self.N[idx] * w[..., None]).sum(1))
        Qp = Q - ((Q - P0) * N0).sum(1, keepdims=True) * N0
        return Qp + N0 * offset, N0

    def opening_edge(self, side, u0, u1):
        m = (self.rside == side) & (self.ru >= u0) & (self.ru <= u1)
        idx = np.nonzero(m)[0]
        idx = idx[np.argsort(self.ru[idx])]
        return self.grid[idx, -1, :], self.ru[idx]

    def collar_curve(self, u_end):
        el, ul = self.opening_edge('lat', 0.0, u_end)
        em, um = self.opening_edge('med', 0.0, u_end)
        hi = np.nonzero(self.rside == 'heel')[0]
        hi = hi[np.argsort(self.rpsi[hi])]
        eh = self.grid[hi, -1, :]
        P = np.concatenate([el[::-1], eh, em], 0)
        uu = np.concatenate([ul[::-1], self.ru[hi], um])
        return P, uu

    def back_line(self):
        hi = np.nonzero(self.rside == 'heel')[0]
        k = hi[np.argmin(np.abs(self.rpsi[hi] - np.pi / 2))]
        return self.grid[k]          # featherline -> edge


# ----------------------------------------------------------------------------
# helper: frames along a curve lying on the upper
# ----------------------------------------------------------------------------

def curve_frames(P, up_hint):
    T = np.gradient(P, axis=0)
    T = norm(T)
    N = norm(up_hint - (up_hint * T).sum(1, keepdims=True) * T)
    B = np.cross(T, N)
    return T, N, B


# ----------------------------------------------------------------------------
# Collar padding (lining roll along the foot opening)
# ----------------------------------------------------------------------------
COLLAR_PROFILE = np.array([
    # (n: outward, v: up) mm  -- closed loop: outer seam -> rolled piping over the top -> inner padding -> back
    (0.30, -1.9), (1.05, -1.2), (1.55, 0.2), (1.45, 1.7), (0.85, 2.9), (-0.30, 3.7), (-1.80, 4.05),
    (-3.50, 3.85), (-5.20, 3.15), (-6.70, 1.85), (-7.70, 0.0), (-8.20, -2.5), (-8.20, -5.5), (-7.60, -8.8),
    (-6.60, -12.2), (-5.40, -15.8), (-4.20, -19.2), (-3.20, -22.5), (-2.60, -24.6), (-2.25, -25.3),
    (-1.80, -24.0), (-1.20, -18.0), (-0.70, -11.0), (-0.35, -6.0), (-0.05, -2.8)])


def collar_mesh(up, u_end=0.372, n_along=260, n_prof=60):
    P, uu = up.collar_curve(u_end)
    P = smooth_poly(P, 1.5)
    P, length = resample_poly(P, n_along)
    s = np.linspace(0, 1, n_along)
    uu = np.interp(s, np.linspace(0, 1, len(uu)), uu)
    _, Nw = up.project(P)
    T = norm(np.gradient(P, axis=0))
    Nw = norm(Nw - (Nw * T).sum(1, keepdims=True) * T)
    Up = np.cross(T, Nw)
    Up *= np.sign(Up[:, 2:3] + 1e-9)
    # surface-following frame below the edge: nearest upper row profile (edge -> down)
    edges = up.grid[:, -1, :]
    etree = cKDTree(edges)
    _, rows = etree.query(P)
    prof = np.vstack([COLLAR_PROFILE, COLLAR_PROFILE[:1]])
    prof, _ = resample_poly(prof, n_prof + 1)
    prof = prof[:-1]
    dist_end = np.minimum(s, 1 - s) * length
    taper = np.clip(dist_end / 0.016, 0, 1) ** 0.6
    heel = np.exp(-((uu - g.UB) / 0.12) ** 2)
    sc_n = (0.62 + 0.38 * heel) * (0.25 + 0.75 * taper)
    sc_v = (0.80 + 0.20 * heel) * (0.35 + 0.65 * taper)
    V = []
    for i in range(n_along):
        n = prof[:, 0] * MM * sc_n[i]
        v = prof[:, 1] * MM * sc_v[i]
        row = up.grid[rows[i]][::-1]            # edge -> featherline
        seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
        sl = np.concatenate([[0], np.cumsum(seg)])
        # shift row so its edge matches P[i]
        shift = P[i] - row[0]
        out = np.zeros((n_prof, 3))
        for j in range(n_prof):
            if v[j] >= 0:
                out[j] = P[i] + n[j] * Nw[i] + v[j] * Up[i]
            else:
                d = min(-v[j], sl[-1])
                S = np.array([np.interp(d, sl, row[:, k]) for k in range(3)]) + shift
                _, NS = up.project(S[None])
                NS = NS[0]
                NS = norm(NS - (NS @ T[i]) * T[i])
                out[j] = S + n[j] * NS
        V.append(out)
    V = np.concatenate(V, 0)
    F = tube_faces(n_along, n_prof, True)
    base = len(V)
    caps = []
    for i, k in ((0, 0), (n_along - 1, 1)):
        ring = np.arange(i * n_prof, i * n_prof + n_prof)
        c = V[ring].mean(0)
        caps.append(c)
        for j in range(n_prof):
            a_, b_ = ring[j], ring[(j + 1) % n_prof]
            F.append((a_, base + k, b_) if k == 0 else (b_, base + k, a_))
    V = np.concatenate([V, np.array(caps)], 0)
    return V, F, dict(P=P, Nw=Nw, Up=Up, uu=uu, s=s)


# ----------------------------------------------------------------------------
# Tongue
# ----------------------------------------------------------------------------

def _arch_drop(p, i, yrel):
    """drop of the last cross-section below its top at lateral offset yrel (m) for station i."""
    out = np.zeros_like(yrel)
    for side, m in (('lat', yrel < 0), ('med', yrel >= 0)):
        if not m.any():
            continue
        phis = np.linspace(0, np.pi / 2, 400)
        y, z = g.arch_point(p, i, side, phis)
        dy = np.abs(y - p['yt'][i])
        o = np.argsort(dy)
        out[m] = p['ztop'][i] - np.interp(np.abs(yrel[m]), dy[o], z[o], right=z[o][-1] - 0.03)
    return out


def tongue_mesh(up, ns=190, nw=64):
    """tongue: in the lacing zone its top follows the last (arch) section, raised inside the throat
    gap and tucked under the knit/eyestay at the sides; above the top crossing it stands free,
    wraps the instep with an arc section and ends in a padded top band."""
    at = g.arch_top_f
    MMl = MM / L
    # centre line of the TOP surface (u, h) -- side photo silhouette
    cl = [(0.726, at(0.726) + 0.5 * MMl), (0.70, at(0.70) + 0.8 * MMl), (0.66, at(0.66) + 0.9 * MMl),
          (0.60, at(0.60) + 1.0 * MMl), (0.54, at(0.54) + 1.1 * MMl), (0.49, at(0.49) + 1.2 * MMl),
          (0.465, 0.4005), (0.450, 0.4110), (0.435, 0.4250), (0.420, 0.4400), (0.407, 0.4515),
          (0.396, 0.4575), (0.387, 0.4590)]
    cl_u = np.array([c[0] for c in cl]); cl_h = np.array([c[1] for c in cl])
    t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cl_u), np.diff(cl_h)))])
    tt_ = np.linspace(0, t[-1], 600)
    xs = PchipInterpolator(t, g.ux(cl_u))(tt_)
    zs = PchipInterpolator(t, cl_h * L)(tt_)
    C = np.stack([xs, g.upper_params(g.xu(xs))['yt'], zs], 1)
    C, length = resample_poly(C, ns)
    s = np.linspace(0, 1, ns)
    uu = g.xu(C[:, 0])
    P = g.upper_params(uu)
    T = norm(np.gradient(C, axis=0))
    Y = np.array([0, 1.0, 0])
    Nn = norm(np.cross(T, Y))
    Nn *= np.sign(Nn[:, 2:3] + 1e-9)
    arc = s * length
    rem = length - arc
    hw = np.interp(uu, [0.395, 0.405, 0.42, 0.44, 0.47, 0.50, 0.60, 0.68, 0.715, 0.738],
                   [15.5, 17.0, 19.0, 21.5, 23.5, 25.0, 24.0, 21.5, 16.0, 10.0]) * MM
    th_c = np.interp(rem, [0.0, 0.003, 0.008, 0.013, 0.020, 0.032, 0.060], [6.2, 6.8, 6.6, 5.4, 3.8, 3.0, 2.6]) * MM
    th_c = np.minimum(th_c, np.interp(arc, [0.0, 0.006, 0.02], [1.0, 2.2, 2.6]) * MM + (arc > 0.02) * 9)
    w_top = g.smoothstep(0.468, 0.442, uu)
    gap = np.array([float(g.throat_f(x)) if 0.355 <= x <= 0.716 else 0.017 for x in uu])
    gap = np.maximum(gap, 0.004)
    tt = np.linspace(-1, 1, nw)
    endcap = np.sqrt(np.clip(1 - ((arc - (length - 0.0034)) / 0.0034).clip(0, 1) ** 2, 0, 1))
    frontcap = np.sqrt(np.clip(1 - ((0.004 - arc) / 0.004).clip(0, 1) ** 2, 0, 1))
    Rarc = 23.0 * MM
    V = []
    for i in range(ns):
        y = tt * hw[i] * (0.7 + 0.3 * frontcap[i])
        ay = np.abs(y)
        lens = np.sqrt(np.clip(1 - tt ** 10, 0, 1))
        # (a) lacing zone
        drop = _arch_drop(P, i, y)
        lift = (C[i, 2] - P['ztop'][i]) * (1 - g.smoothstep(0.55 * gap[i], 1.15 * gap[i], ay)) \
            - 2.7 * MM * g.smoothstep(0.80 * gap[i], 1.25 * gap[i], ay)
        top_a = P['ztop'][i] - drop + lift - C[i, 2]
        th_a = th_c[i] * (1 - 0.35 * g.smoothstep(0.6 * gap[i], 1.3 * gap[i], ay))
        # (b) free part: arc section wrapping the instep
        top_b = -(Rarc - np.sqrt(np.clip(Rarc ** 2 - ay ** 2, 0, None)))
        wt = w_top[i]
        cap = max(endcap[i], 0.03) * max(frontcap[i], 0.05)
        thick = ((1 - wt) * th_a + wt * th_c[i]) * lens * cap
        offs_top = (1 - wt) * top_a + wt * top_b
        top = C[i] + y[:, None] * Y + offs_top[:, None] * Nn[i]
        bot = top - thick[:, None] * Nn[i]
        V.append(np.concatenate([top, bot[::-1][1:-1]], 0))
    nr = V[0].shape[0]
    V = np.concatenate(V, 0)
    F = tube_faces(ns, nr, True)
    Vw, Fw, inv = g.weld(V, F, 2e-7)
    sa = np.repeat(s, nr)
    ta = np.tile(np.concatenate([tt, tt[::-1][1:-1]]), ns)
    fr = np.tile(np.concatenate([np.ones(nw), np.zeros(nw - 2)]), ns)
    arc_a = np.repeat(rem, nr)                       # distance from the top edge (m)
    ty = np.concatenate([np.outer(np.ones(1), tt * hw[i] * (0.7 + 0.3 * frontcap[i]))[0] for i in range(ns)])
    ty = np.concatenate([np.concatenate([ty[i * nw:(i + 1) * nw], ty[i * nw:(i + 1) * nw][::-1][1:-1]]) for i in range(ns)])
    A = {}
    for k, a_ in (('ts', sa), ('tt', ta), ('tfront', fr), ('tarc', arc_a), ('ty', ty)):
        b_ = np.zeros(len(Vw)); b_[inv] = a_; A[k] = b_
    return Vw, Fw, A, dict(C=C, Nn=Nn, hw=hw, th=th_c, s=s, uu=uu, length=length, arc=arc, rem=rem)


def tongue_top_surface(tg):
    """function (x,y) -> z of the tongue top surface (for lace placement)."""
    C, Nn, hw, th, s, sag = tg['C'], tg['Nn'], tg['hw'], tg['th'], tg['s'], tg['sag']
    pts = []
    for i in range(len(C)):
        tt = np.linspace(-1.3, 1.3, 41)
        y = tt * hw[i]
        drop = -sag[i] * tt ** 2
        P = C[i] + y[:, None] * np.array([0, 1.0, 0]) + (drop + th[i] / 2)[:, None] * Nn[i]
        pts.append(P)
    P = np.concatenate(pts)
    tree = cKDTree(P[:, :2])

    def zf(x, y):
        d, idx = tree.query(np.stack([x, y], -1), k=3)
        w = 1 / np.maximum(d, 1e-6)
        return (P[idx, 2] * w).sum(-1) / w.sum(-1)
    return zf


# ----------------------------------------------------------------------------
# Eyestay overlay (TPU) + eyelets
# ----------------------------------------------------------------------------
# measured on the studio photos (u along length, h = height / L)
EYELETS = {'lat': [(0.435, 0.358), (0.488, 0.347), (0.545, 0.322), (0.597, 0.300), (0.650, 0.276), (0.700, 0.252)],
           'med': [(0.450, 0.355), (0.502, 0.347), (0.558, 0.328), (0.609, 0.305), (0.660, 0.283), (0.715, 0.257)]}
_EB = json.load(open(os.path.join(HERE, 'ref', 'eyestay_bounds.json')))
ES_LOW = {k: (np.array([p[0] for p in v]), gaussian_filter1d(np.array([p[1] for p in v]), 1.0)) for k, v in _EB.items()}


def side_point(up, side, u, h):
    """point on the upper's outer wall at length u and height h (L units), seen from that side."""
    u = np.atleast_1d(u).astype(float); h = np.atleast_1d(h).astype(float)
    x = g.ux(u); z = h * L
    yt = np.interp(u, up.u, up.p['yt'])
    out = np.zeros((len(u), 3)); nrm = np.zeros((len(u), 3))
    for it, (span, n) in enumerate(((0.075, 151), (0.0012, 49))):
        if it == 0:
            ys = np.linspace(0, span, n)[None, :] * (-1 if side == 'lat' else 1) + yt[:, None]
        else:
            ys = out[:, 1:2] + np.linspace(-span, span, n)[None, :]
        Q = np.stack([np.repeat(x[:, None], ys.shape[1], 1), ys, np.repeat(z[:, None], ys.shape[1], 1)], -1)
        d, _ = up.tree.query(Q.reshape(-1, 3))
        d = d.reshape(Q.shape[:2])
        # outermost local minimum close to the surface
        j = np.argmin(d + 0.0 * ys, axis=1)
        out = Q[np.arange(len(u)), j]
    P, N = up.project(out)
    return P, N


def eyestay_mesh(up, n_along=360, n_across=28):
    el, ul = up.opening_edge('lat', 0.355, 0.716)
    em, um = up.opening_edge('med', 0.368, 0.716)
    E = np.concatenate([el, em[::-1][1:]], 0)
    uE = np.concatenate([ul, um[::-1][1:]])
    sideE = np.concatenate([np.zeros(len(ul)), np.ones(len(um) - 1)])
    E = smooth_poly(E, 2.0)
    E, length = resample_poly(E, n_along)
    s = np.linspace(0, 1, n_along)
    src_s = np.linspace(0, 1, len(uE))
    uE = np.interp(s, src_s, uE)
    sideE = np.interp(s, src_s, sideE)
    _, N = up.project(E)
    T = norm(np.gradient(E, axis=0))
    D = norm(np.cross(N, T))
    centre = np.stack([E[:, 0], np.interp(uE, up.u, up.p['yt']), E[:, 2]], 1)
    away = E - centre
    ref = (uE > 0.45) & (uE < 0.62)
    D *= np.sign(np.median((D[ref] * away[ref]).sum(1)))
    # lower boundary: measured side-view curve (per side) blended into a front U band
    low = np.zeros_like(E)
    for side, m in (('lat', sideE < 0.5), ('med', sideE >= 0.5)):
        uu_, hh_ = ES_LOW[side]
        uq = np.clip(uE[m], uu_[0], uu_[-1])
        hq = np.interp(uq, uu_, hh_)
        P, _ = side_point(up, side, uq, hq)
        low[m] = P
    w_front = 11.0 * MM
    front = E + D * w_front
    wf = g.smoothstep(0.655, 0.705, uE)
    low = low * (1 - wf[:, None]) + front * wf[:, None]
    low = smooth_poly(low, 2.0)
    # never narrower than 4 mm, rounded start caps
    dist_end = np.minimum(s, 1 - s) * length
    cap = np.clip(dist_end / 0.006, 0, 1) ** 0.5
    vec = low - E
    wid = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.maximum(wid, 1e-9) * np.maximum(wid, 4.0 * MM)
    low = E + vec * (0.25 + 0.75 * cap)[:, None]
    tt = np.linspace(0, 1, n_across)
    V = (E[:, None, :] * (1 - tt[None, :, None]) + low[:, None, :] * tt[None, :, None]).reshape(-1, 3)
    # small overlap over the knit edge
    V = V - (D[:, None, :] * (0.4 * MM) * (tt[None, :, None] == 0)).reshape(-1, 3)
    Vp, Np = up.project(V, offset=0.00045)
    F = tube_faces(n_along, n_across, closed_ring=False)
    sa = np.repeat(s, n_across)
    ta = np.tile(tt, n_along)
    return Vp, F, dict(E=E, D=D, N=N, uE=uE, s=s, side=sideE, sa=sa, ta=ta, length=length, low=low)


def eyelet_points(up, es):
    out = []
    for side in ('lat', 'med'):
        for k, (u, h) in enumerate(EYELETS[side]):
            P, N = side_point(up, side, u, h)
            m = es['side'] < 0.5 if side == 'lat' else es['side'] > 0.5
            idx = np.nonzero(m)[0]
            j = idx[np.argmin(np.linalg.norm(es['E'][idx] - P[0], axis=1))]
            T = norm(np.gradient(es['E'], axis=0))[j]
            out.append(dict(side=side, k=k, P=P[0], N=N[0], T=T, D=es['D'][j]))
    return out


# ----------------------------------------------------------------------------
# Laces
# ----------------------------------------------------------------------------
LACE_W, LACE_T = 6.9 * MM, 2.5 * MM


def lace_segments(eyes):
    E = {(e['side'], e['k']): e for e in eyes}
    segs = [((('lat', 5), ('med', 5)), 0)]          # bottom bar
    for k in range(5, 0, -1):
        over = (k % 2)
        segs.append(((('lat', k), ('med', k - 1)), over))
        segs.append(((('med', k), ('lat', k - 1)), 1 - over))
    return [(E[a], E[b], o) for (a, b), o in segs]


def lace_mesh(eyes, surf_z, n_along=110, n_prof=18):
    """surf_z(x, y) -> top surface height (eyestay/tongue/upper) under the lace."""
    segs = lace_segments(eyes)
    Vs, Fs, UVs = [], [], []
    base = 0
    for si, (A, B, over) in enumerate(segs):
        t = np.linspace(0, 1, n_along)
        PA, PB = np.asarray(A['P']), np.asarray(B['P'])
        NA, NB = np.asarray(A['N']), np.asarray(B['N'])
        xy = PA[None, :2] * (1 - t[:, None]) + PB[None, :2] * t[:, None]
        zs = surf_z(xy[:, 0], xy[:, 1])
        bulge = (0.35 + 0.9 * np.sin(np.pi * t)) * MM
        cross = 1.9 * MM * np.exp(-((t - 0.5) / 0.13) ** 2) * (1 if over else 0)
        z = zs + LACE_T / 2 + bulge + cross
        P = np.stack([xy[:, 0], xy[:, 1], z], 1)
        P = smooth_poly(P, 3.0)
        seglen0 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        total = seglen0[-1]
        # dive into the eyelet holes along -normal: only inside the hole radius (~2 mm)
        for end, Pe, Ne in ((0, PA, NA), (1, PB, NB)):
            d = seglen0 if end == 0 else total - seglen0
            w = 1 - g.smoothstep(0.0, 2.6 * MM, d)
            target = Pe[None] + Ne[None] * (1.35 * MM - 3.2 * MM * (1 - d / (2.6 * MM)).clip(0, 1)[:, None])
            P = P * (1 - w[:, None]) + target * w[:, None]
        T = norm(np.gradient(P, axis=0))
        upv = np.array([0, 0, 1.0])[None].repeat(n_along, 0)
        wA = np.exp(-(t / 0.14) ** 2)[:, None]
        wB = np.exp(-((1 - t) / 0.14) ** 2)[:, None]
        upv = norm(upv * np.clip(1 - wA - wB, 0, 1) + NA[None] * wA + NB[None] * wB)
        Nn = norm(upv - (upv * T).sum(1, keepdims=True) * T)
        Bn = np.cross(T, Nn)
        a_ = np.linspace(0, 2 * np.pi, n_prof, endpoint=False)
        px = np.cos(a_) * LACE_W / 2
        py = np.sign(np.sin(a_)) * np.abs(np.sin(a_)) ** 0.75 * LACE_T / 2
        dA = seglen0; dB = total - seglen0
        endw = (1 - g.smoothstep(0.5 * MM, 6.0 * MM, dA)) + (1 - g.smoothstep(0.5 * MM, 6.0 * MM, dB))
        wsc = 1 - 0.5 * np.clip(endw, 0, 1)
        V = (P[:, None, :] + (px[None, :, None] * wsc[:, None, None]) * Bn[:, None, :] + py[None, :, None] * Nn[:, None, :])
        V = V.reshape(-1, 3)
        F = [tuple(x + base for x in f) for f in tube_faces(n_along, n_prof, True)]
        Vs.append(V); Fs += F
        seglen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        uvu = np.repeat(seglen / 0.004, n_prof)
        uvv = np.tile(a_ / (2 * np.pi), n_along)
        UVs.append(np.stack([uvu, uvv], 1))
        base += len(V)
    return np.concatenate(Vs), Fs, np.concatenate(UVs)


# ----------------------------------------------------------------------------
# Heel pull tab
# ----------------------------------------------------------------------------

def heel_tab_mesh(up, collar, n_along=160, width=10.5 * MM, thick=1.7 * MM):
    # heel back center line of the upper (station 0)
    back = up.back_line()[::-1]                      # edge -> featherline
    back = back[np.argsort(-back[:, 2])]
    zb_top = back[0, 2]
    xb_top = back[0, 0]
    # collar roll top at the back
    i0 = np.argmin(np.abs(collar['uu'] - g.UB))
    roll_top = collar['P'][i0] + collar['Up'][i0] * 0.0033
    pts = np.array([
        (xb_top + 2.8 * MM, zb_top - 12 * MM), (xb_top + 2.4 * MM, zb_top - 4 * MM),
        (xb_top + 1.6 * MM, zb_top + 2.5 * MM), (xb_top + 0.4 * MM, zb_top + 6.2 * MM),
        (xb_top - 1.2 * MM, zb_top + 8.4 * MM), (xb_top - 3.0 * MM, zb_top + 8.9 * MM),
        (xb_top - 4.6 * MM, zb_top + 7.6 * MM), (xb_top - 5.1 * MM, zb_top + 4.8 * MM),
        (xb_top - 4.4 * MM, zb_top + 1.5 * MM), (xb_top - 3.0 * MM, zb_top - 1.5 * MM),
        (xb_top - 2.0 * MM, zb_top - 4.5 * MM)])
    # follow the heel counter down to h=0.19 L, 1.1 mm proud of the surface
    zz = np.linspace(zb_top - 5.5 * MM, 0.192 * L, 40)
    xx = np.interp(zz, back[::-1, 2], back[::-1, 0]) - 1.1 * MM
    tail = np.stack([xx, zz], 1)
    P2 = np.concatenate([pts, tail], 0)
    # smooth spline through
    tparam = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P2, axis=0), axis=1))])
    cs = CubicSpline(tparam, P2, axis=0)
    tt = np.linspace(0, tparam[-1], n_along)
    P = cs(tt)
    P = np.stack([P[:, 0], np.zeros(n_along), P[:, 1]], 1)
    T = norm(np.gradient(P, axis=0))
    Y = np.array([0, 1.0, 0])
    Nn = norm(np.cross(T, Y))
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    px = np.cos(a) * width / 2
    py = np.sign(np.sin(a)) * np.abs(np.sin(a)) ** 0.4 * thick / 2
    V = P[:, None, :] + px[None, :, None] * Y[None, None, :] + py[None, :, None] * Nn[:, None, :]
    V = V.reshape(-1, 3)
    F = tube_faces(n_along, len(a), True)
    # material: side of the strap facing +Nn vs -Nn
    side = np.tile((np.sin(a) > 0).astype(float), n_along)
    # end caps
    base = len(V)
    caps = []
    for i, k in ((0, 0), (n_along - 1, 1)):
        ring = np.arange(i * len(a), (i + 1) * len(a))
        caps.append(V[ring].mean(0))
        for j in range(len(a)):
            x, y = ring[j], ring[(j + 1) % len(a)]
            F.append((x, base + k, y) if k == 0 else (y, base + k, x))
    V = np.concatenate([V, np.array(caps)])
    side = np.concatenate([side, [0, 0]])
    return V, F, side, Nn


# ----------------------------------------------------------------------------
# Insole
# ----------------------------------------------------------------------------

def insole_mesh(up, nu=200, nw=40):
    s = np.linspace(0, 1, nu)
    u = g.UB + 0.004 + (g.UF - 0.03 - g.UB) * (0.5 - 0.5 * np.cos(np.pi * s))
    p = g.upper_params(u)
    inset = 2.6 * MM
    al = np.maximum(p['al'] - inset, 0)
    am = np.maximum(p['am'] - inset, 0)
    zl = p['zfl'] + 4.5 * MM - 0.6 * MM
    zm = p['zfm'] + 4.5 * MM - 0.6 * MM
    V, UV = [], []
    tt = np.linspace(-1, 1, nw)
    for i in range(nu):
        y = np.where(tt < 0, tt * al[i], tt * am[i]) + 0 * p['yt'][i]
        zedge = np.where(tt < 0, zl[i], zm[i])
        zc = 0.5 * (zl[i] + zm[i]) - 1.8 * MM
        z = zc + (zedge - zc) * np.abs(tt) ** 2.5 + 1.2 * MM * np.abs(tt) ** 12
        V.append(np.stack([np.full(nw, g.ux(u[i])), y, z], 1))
    V = np.concatenate(V, 0)
    F = tube_faces(nu, nw, closed_ring=False)
    Vw, Fw, inv = g.weld(V, F, 1e-7)
    return Vw, Fw


# ----------------------------------------------------------------------------
# Side badge (raised logo) on the lateral wall
# ----------------------------------------------------------------------------

def lateral_wall_y(x, z, up):
    """y on the upper outer surface for given (x,z) on the lateral side (ray along +y)."""
    # use dense ray-march on the KD-tree: start far out and search nearest along y
    out = np.zeros_like(x)
    ys = np.linspace(-0.07, 0.0, 141)
    for k in range(len(x)):
        Q = np.stack([np.full_like(ys, x[k]), ys, np.full_like(ys, z[k])], 1)
        d, _ = up.tree.query(Q)
        j = np.argmin(d)
        out[k] = ys[j]
    return out


def piping_mesh(up, n_along=220, n_prof=12, r=1.45 * MM):
    """rolled binding along the throat edge (continuation of the collar lining)."""
    Vs, Fs = [], []
    base = 0
    for side in ('lat', 'med'):
        e, u = up.opening_edge(side, 0.356, 0.70)
        P = smooth_poly(e, 1.5)
        P, _ = resample_poly(P, n_along)
        _, N = up.project(P)
        T = norm(np.gradient(P, axis=0))
        B = norm(np.cross(T, N))
        Up_ = B * np.sign((B[:, 2:3]) + 1e-9)
        a = np.linspace(0, 2 * np.pi, n_prof, endpoint=False)
        C = P + N * (0.9 * MM) + Up_ * (0.2 * MM)
        V = C[:, None, :] + r * (np.cos(a)[None, :, None] * N[:, None, :] + np.sin(a)[None, :, None] * Up_[:, None, :])
        Vs.append(V.reshape(-1, 3))
        Fs += [tuple(x + base for x in f) for f in tube_faces(n_along, n_prof, True)]
        base += n_along * n_prof
    return np.concatenate(Vs), Fs
