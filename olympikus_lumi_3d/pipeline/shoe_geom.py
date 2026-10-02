"""Parametric geometry for the Olympikus Lumi digital twin.

All shape curves come from spec.json (measured on the studio reference photos).
Units: meters. Shoe axis: +X toe, -X heel, +Y medial, -Y lateral (right shoe), Z up.
u in [0,1] is the normalized length coordinate (0 = heel back, 1 = toe tip).
"""
import json, os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'spec.json')))
L = 0.265                      # outsole length (m)
UU = np.array(S['u'])
X0 = -L / 2                    # x of the heel back (model centered on X)


def _curve(name, smooth=0.0):
    a = np.array(S[name], float)
    if smooth:
        a = gaussian_filter1d(a, smooth)
    return lambda u: np.interp(u, UU, a)


def pchip(pts):
    pts = np.array(pts, float)
    f = PchipInterpolator(pts[:, 0], pts[:, 1], extrapolate=False)
    lo, hi = pts[0, 0], pts[-1, 0]
    return lambda u: f(np.clip(u, lo, hi))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


w_lat_raw = _curve('w_lat', 1.5)
w_med_raw = _curve('w_med', 1.5)
mid_lat = _curve('mid_lat', 2)
mid_med = _curve('mid_med', 2)
bot_env = _curve('bot_env', 2)
top_lat = _curve('top_lat', 1)
top_med = _curve('top_med', 1)


def ux(u):
    return X0 + u * L


def xu(x):
    return (x - X0) / L


# ----------------------------------------------------------------------------
# Footprint (plan view) of the midsole: half widths in L units
# ----------------------------------------------------------------------------
WIDTH_SCALE = 1.0    # top-view widths match after perspective-correct comparison


def footprint(u):
    u = np.atleast_1d(np.asarray(u, float))
    wl = w_lat_raw(u) * WIDTH_SCALE
    wm = w_med_raw(u) * WIDTH_SCALE
    # heel closure (round heel) for u<0.03 ; toe closure for u>0.985
    for (ua, ub, side) in [(0.0, 0.03, 'heel'), (1.0, 0.985, 'toe')]:
        if side == 'heel':
            m = u < ub
            t = np.clip((u[m] - ua) / (ub - ua), 0, 1)
            f = np.sqrt(np.clip(t * (2 - t), 0, 1))
            wl[m] = w_lat_raw(ub) * WIDTH_SCALE * f
            wm[m] = w_med_raw(ub) * WIDTH_SCALE * f
        else:
            m = u > ub
            t = np.clip((ua - u[m]) / (ua - ub), 0, 1)
            f = np.sqrt(np.clip(t * (2 - t), 0, 1))
            wl[m] = w_lat_raw(ub) * WIDTH_SCALE * f
            wm[m] = w_med_raw(ub) * WIDTH_SCALE * f
    return wl, wm


# ----------------------------------------------------------------------------
# Midsole sidewall sculpt: crease curves (u, h) measured from photo shading
# ----------------------------------------------------------------------------
CREASES = {
    'lat': [
        # (points, depth_mm, width_above_mm, width_below_mm)
        ([(0.07, 0.170), (0.11, 0.161), (0.18, 0.152), (0.25, 0.147), (0.33, 0.143), (0.42, 0.134),
          (0.51, 0.116), (0.60, 0.096), (0.69, 0.086), (0.78, 0.083), (0.86, 0.104), (0.92, 0.128),
          (0.955, 0.150)], 1.4, 1.3, 4.5),
        ([(0.030, 0.186), (0.045, 0.176), (0.09, 0.150), (0.13, 0.133), (0.23, 0.106), (0.32, 0.093),
          (0.41, 0.100), (0.50, 0.095), (0.59, 0.073), (0.68, 0.060), (0.77, 0.058), (0.84, 0.066)],
         2.4, 1.6, 6.0),
        ([(0.005, 0.141), (0.04, 0.125), (0.06, 0.111), (0.08, 0.100), (0.11, 0.092), (0.13, 0.089),
          (0.17, 0.091), (0.21, 0.095), (0.26, 0.098)], 1.9, 1.3, 5.0),
        ([(0.012, 0.085), (0.03, 0.076), (0.05, 0.064), (0.07, 0.052), (0.10, 0.039), (0.13, 0.030),
          (0.16, 0.024)], 1.5, 1.2, 4.0),
        ([(0.21, 0.066), (0.25, 0.065), (0.29, 0.057), (0.33, 0.046), (0.37, 0.034), (0.40, 0.026)],
         1.2, 1.2, 4.0),
        ([(0.33, 0.079), (0.37, 0.071), (0.41, 0.061), (0.45, 0.053), (0.50, 0.043), (0.56, 0.036),
          (0.61, 0.038), (0.65, 0.043)], 1.3, 1.2, 4.0),
        ([(0.72, 0.040), (0.76, 0.050), (0.81, 0.062), (0.85, 0.077), (0.89, 0.094), (0.93, 0.115)],
         1.2, 1.2, 4.0),
    ],
    'med': [
        ([(0.07, 0.172), (0.12, 0.160), (0.17, 0.152), (0.23, 0.149), (0.30, 0.152), (0.35, 0.153),
          (0.42, 0.145), (0.50, 0.129), (0.57, 0.114), (0.65, 0.102), (0.73, 0.095), (0.80, 0.096),
          (0.88, 0.113), (0.95, 0.145)], 1.4, 1.3, 4.5),
        ([(0.030, 0.182), (0.08, 0.150), (0.12, 0.134), (0.15, 0.131), (0.22, 0.109), (0.30, 0.094),
          (0.38, 0.100), (0.46, 0.110), (0.54, 0.097), (0.62, 0.081), (0.70, 0.072), (0.78, 0.072),
          (0.84, 0.078)], 2.4, 1.6, 6.0),
        ([(0.005, 0.141), (0.04, 0.128), (0.06, 0.117), (0.08, 0.104), (0.10, 0.096), (0.12, 0.094),
          (0.15, 0.089), (0.20, 0.094), (0.25, 0.098)], 1.9, 1.3, 5.0),
        ([(0.012, 0.080), (0.04, 0.064), (0.07, 0.044), (0.09, 0.030), (0.12, 0.020), (0.15, 0.015)],
         1.5, 1.2, 4.0),
        ([(0.22, 0.066), (0.26, 0.063), (0.29, 0.060), (0.33, 0.052), (0.36, 0.043), (0.39, 0.033)],
         1.2, 1.2, 4.0),
        ([(0.35, 0.081), (0.38, 0.076), (0.42, 0.070), (0.47, 0.060), (0.51, 0.050), (0.57, 0.045),
          (0.63, 0.051), (0.67, 0.057)], 1.3, 1.2, 4.0),
        ([(0.76, 0.058), (0.80, 0.069), (0.84, 0.081), (0.89, 0.099), (0.94, 0.119), (0.965, 0.132)],
         1.2, 1.2, 4.0),
    ],
}


def _crease_disp(u, z, side):
    """inward displacement (m) of the sidewall at (u, z) for one side"""
    d = np.zeros_like(z)
    for pts, depth, wa, wb in CREASES[side]:
        pts = np.array(pts)
        f = pchip(pts)
        zk = f(u) * L
        # fade in/out at the ends of the crease
        u0, u1 = pts[0, 0], pts[-1, 0]
        span = max(u1 - u0, 1e-3)
        fade = smoothstep(u0 - 0.02, u0 + min(0.05, span * 0.3), u) * \
            (1 - smoothstep(u1 - min(0.05, span * 0.3), u1 + 0.02, u))
        # heel-back creases continue around the heel (u<u0 near heel): keep them for heel ones
        if u0 < 0.035:
            fade = np.where(u < u0, 1.0, fade)
        t = (z - zk)
        g = np.where(t > 0, np.exp(-(t / (wa * 1e-3)) ** 2), np.exp(-(t / (wb * 1e-3)) ** 2))
        d += depth * 1e-3 * g * fade
    return d


def crease_disp(u, z, y):
    wl = smoothstep(0.004, -0.004, y)   # 1 on lateral side
    return wl * _crease_disp(u, z, 'lat') + (1 - wl) * _crease_disp(u, z, 'med')


# ----------------------------------------------------------------------------
# Midsole
# ----------------------------------------------------------------------------

def _resample(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return np.repeat(poly[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(poly.shape[1])], 1)


def _round_corner(pts, k, r, n=10):
    """replace corner pts[k] by a quadratic bezier with legs r (polyline densely sampled)"""
    P = np.asarray(pts, float)
    C = P[k]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    sk = s[k]
    r = min(r, 0.45 * sk, 0.45 * (s[-1] - sk))
    if r < 1e-6:
        return P
    sa, sb = sk - r, sk + r
    pa = np.array([np.interp(sa, s, P[:, j]) for j in range(P.shape[1])])
    pb = np.array([np.interp(sb, s, P[:, j]) for j in range(P.shape[1])])
    t = np.linspace(0, 1, n)[:, None]
    B = (1 - t) ** 2 * pa + 2 * t * (1 - t) * C + t ** 2 * pb
    keep_a = P[s < sa]
    keep_b = P[s > sb]
    return np.concatenate([keep_a, B, keep_b], 0)


_OUTLINE = None


def upper_outline(n=900):
    """plan-view featherline of the upper: closed polyline (x,y) + rim height z (m)."""
    global _OUTLINE
    if _OUTLINE is not None:
        return _OUTLINE
    s = np.linspace(0, 1, n)
    u = UB + (UF - UB) * (0.5 - 0.5 * np.cos(np.pi * s))
    p = upper_params(u)
    lat = np.stack([ux(u), -p['al'], mid_lat(u) * L], 1)
    med = np.stack([ux(u), p['am'], mid_med(u) * L], 1)
    ring = np.concatenate([lat, med[::-1][1:-1]], 0)
    # smooth rim height around the perimeter (lateral/medial heights differ at heel & toe)
    ring[:, 2] = gaussian_filter1d(ring[:, 2], 60, mode='wrap')
    from scipy.spatial import cKDTree
    _OUTLINE = (ring, cKDTree(ring[:, :2]), u, p)
    return _OUTLINE


HEEL_A, HEEL_B = 0.046 * L, 0.090 * L   # elliptical heel/toe bumper roll-off


def midsole_top(x, y):
    """height of the midsole top surface at plan point (x,y)."""
    ring, tree, uo, p = upper_outline()
    x = np.asarray(x, float); y = np.asarray(y, float)
    d, idx = tree.query(np.stack([x, y], -1))
    zr = ring[idx, 2]
    u = xu(x)
    al = np.interp(u, uo, p['al'], left=0, right=0)
    am = np.interp(u, uo, p['am'], left=0, right=0)
    inside = (u > UB) & (u < UF) & (y > -al) & (y < am)
    dd = np.minimum(d, HEEL_A * 0.999)
    drop = HEEL_B * (1 - np.sqrt(1 - (dd / HEEL_A) ** 2))
    drop = np.where(d >= HEEL_A, HEEL_B + (d - HEEL_A) * 3, drop)
    z_out = zr - drop
    # inside: rim continues ~1.5mm then dips to the footbed (6mm lower)
    z_in = zr - 0.006 * smoothstep(0.0015, 0.007, d)
    return np.where(inside, z_in, z_out)


def midsole_mesh(nu=380, nh=84):
    """Structured ring mesh. Ring order: lateral half (top-center -> bottom-center),
    then medial half (bottom-center -> top-center)."""
    s = np.linspace(0, 1, nu)
    u = 0.5 - 0.5 * np.cos(np.pi * s)
    u = 0.55 * u + 0.45 * s
    wl, wm = footprint(u)
    zb = bot_env(u) * L
    rings = []
    for i in range(nu):
        x = ux(u[i])
        halves = []
        for W, sgn in [(wl[i] * L, -1), (wm[i] * L, 1)]:
            W = max(W, 1e-6)
            ntop = 26
            yy = W * (1 - (1 - np.linspace(0, 1, ntop)) ** 1.3)
            zt = midsole_top(np.full(ntop, x), sgn * yy)
            zt = np.maximum(zt, zb[i] + 0.0012)
            H = zt[-1] - zb[i]
            rb = min(0.0085, 0.45 * H, 0.45 * W)
            rt = min(0.0022, 0.30 * H, 0.30 * W)
            top = np.stack([yy, zt], 1)
            nw = 24
            wall = np.stack([np.full(nw, W), np.linspace(zt[-1], zb[i], nw)], 1)[1:]
            nb = 18
            bot = np.stack([np.linspace(W, 0, nb), np.full(nb, zb[i])], 1)[1:]
            poly = np.concatenate([top, wall, bot], 0)
            k_top = ntop - 1
            k_bot = ntop - 1 + nw - 1
            poly = _round_corner(poly, k_bot, rb, 16)
            poly = _round_corner(poly, k_top, rt, 8)
            hp = _resample(poly, nh)
            hp[:, 0] *= sgn
            halves.append(hp)
        ring = np.concatenate([halves[0], halves[1][::-1][1:-1]], 0)
        rings.append(ring)
    nr = rings[0].shape[0]
    V = np.zeros((nu, nr, 3))
    for i in range(nu):
        V[i, :, 0] = ux(u[i])
        V[i, :, 1] = rings[i][:, 0]
        V[i, :, 2] = rings[i][:, 1]
    return u, V


def grid_faces(nu, nr, closed_ring=True):
    F = []
    for i in range(nu - 1):
        for j in range(nr if closed_ring else nr - 1):
            j2 = (j + 1) % nr
            a, b, c, d = i * nr + j, i * nr + j2, (i + 1) * nr + j2, (i + 1) * nr + j
            F.append((a, b, c, d))
    return np.array(F)


def weld(V, F, tol=2e-6):
    """merge coincident vertices; returns V2, faces (list of tuples, degenerate removed), index map"""
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    V2 = V[first]
    out = []
    for f in F:
        g = [int(inv[i]) for i in f]
        uniq = []
        for v in g:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) >= 3:
            out.append(tuple(uniq))
    return V2, out, inv


def vertex_normals_poly(V, F):
    n = np.zeros_like(V)
    for f in F:
        P = V[list(f)]
        fn = np.zeros(3)
        for k in range(1, len(f) - 1):
            fn += np.cross(P[k] - P[0], P[k + 1] - P[0])
        for v in f:
            n[v] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def vertex_normals(V, F):
    P = V
    n = np.zeros_like(P)
    F = np.asarray(F)
    for tri in ([0, 1, 2], [0, 2, 3]):
        a, b, c = P[F[:, tri[0]]], P[F[:, tri[1]]], P[F[:, tri[2]]]
        fn = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(n, F[:, tri[k]], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def build_midsole():
    u, V = midsole_mesh()
    nu, nr, _ = V.shape
    Vf = V.reshape(-1, 3).copy()
    F = grid_faces(nu, nr, True)
    uu = np.repeat(u, nr)
    Vw, Fw, inv = weld(Vf, F)
    uw = np.zeros(len(Vw)); uw[inv] = uu
    N = vertex_normals_poly(Vw, Fw)
    nh = N.copy()
    nh[:, 2] = 0
    nl = np.linalg.norm(nh, axis=1, keepdims=True)
    nh = nh / np.maximum(nl, 1e-9)
    d = crease_disp(uw, Vw[:, 2], Vw[:, 1])
    zb = bot_env(uw) * L
    ztop = midsole_top(Vw[:, 0], Vw[:, 1])
    wallw = smoothstep(zb + 0.0025, zb + 0.009, Vw[:, 2]) * (1 - smoothstep(ztop - 0.003, ztop - 0.0006, Vw[:, 2]))
    wallw *= smoothstep(0.45, 0.85, nl[:, 0])
    Vd = Vw - nh * (d * wallw)[:, None]
    return Vd, Fw, uw


# ----------------------------------------------------------------------------
# Upper (knit) : analytic last with collar + throat cut
# ----------------------------------------------------------------------------
UB = 0.048     # heel back of the upper (u)
UF = 0.982     # toe tip of the upper (u)
INSET = 0.0048  # upper featherline inset from midsole outline (m)

# collar line height (L units), heel tab excluded
collar_h = pchip([(0.040, 0.392), (0.060, 0.397), (0.085, 0.397), (0.11, 0.391), (0.135, 0.376),
                  (0.16, 0.356), (0.19, 0.336), (0.22, 0.324), (0.25, 0.3195), (0.28, 0.324),
                  (0.31, 0.336), (0.335, 0.352), (0.355, 0.370), (0.372, 0.392), (0.39, 0.43),
                  (0.42, 0.52)])
# arch (last) top height for the closed/instep part (L units)
arch_top_pts = [(0.30, 0.452), (0.36, 0.430), (0.40, 0.418), (0.45, 0.405), (0.50, 0.384),
                (0.55, 0.356), (0.60, 0.331), (0.65, 0.298), (0.70, 0.277), (0.75, 0.254),
                (0.80, 0.240), (0.85, 0.235), (0.90, 0.2345), (0.93, 0.230), (0.955, 0.219),
                (0.97, 0.205), (0.982, 0.188)]
arch_top_f = pchip(arch_top_pts)
# throat half gap (m) measured from the arch top line
throat_pts = [(0.355, 0.0), (0.366, 0.0160), (0.39, 0.0179), (0.45, 0.0172), (0.54, 0.0158),
              (0.62, 0.0148), (0.67, 0.0136), (0.692, 0.0116), (0.705, 0.0080), (0.712, 0.0042),
              (0.716, 0.0)]
throat_f = pchip(throat_pts)
N_EXP = 2.4       # superellipse exponent of the last cross-section


def upper_params(u):
    u = np.atleast_1d(np.asarray(u, float))
    wl, wm = footprint(u)
    al = wl * L - INSET
    am = wm * L - INSET
    # heel cap of the upper (it sits ~13 mm in front of the midsole heel bulge)
    uc = UB + 0.115
    m = u < uc
    t = np.clip((u[m] - UB) / (uc - UB), 0, 1)
    f = np.clip(t * (2 - t), 0, 1) ** 0.5
    al[m] *= f
    am[m] *= f
    # toe cap
    utc = UF - 0.06
    m = u > utc
    t = np.clip((UF - u[m]) / (UF - utc), 0, 1)
    f = np.clip(t * (2 - t), 0, 1) ** 0.5
    al[m] *= f
    am[m] *= f
    al = np.maximum(al, 0)
    am = np.maximum(am, 0)
    zfl = mid_lat(u) * L - 0.0045
    zfm = mid_med(u) * L - 0.0045
    wend = smoothstep(UB, UB + 0.05, u) * (1 - smoothstep(UF - 0.05, UF, u))
    zavg = 0.5 * (zfl + zfm)
    zfl = zavg + (zfl - zavg) * wend
    zfm = zavg + (zfm - zavg) * wend
    # arch top
    zc = collar_h(u) * L - 0.0036   # knit edge sits below the padded roll
    at = arch_top_f(u) * L
    # in the heel region the arch top is above the collar; lean-in of the walls at the collar
    lean = 0.0090
    zf = 0.5 * (zfl + zfm)
    a = np.maximum(0.5 * (al + am), 1e-4)
    frac = np.clip(1 - lean / a, 0.05, 0.995)
    nexp0 = N_EXP + 0.65 * smoothstep(0.26, 0.40, u) * (1 - smoothstep(0.66, 0.84, u))
    fh = (1 - frac ** nexp0) ** (1 / nexp0)
    H_heel = (zc - zf) / np.maximum(fh, 0.05)
    top_heel = zf + H_heel
    wblend = smoothstep(0.30, 0.40, u)
    ztop = (1 - wblend) * top_heel + wblend * at
    yt = 0.3 * (am - al)          # top center line y (lacing centre ~4 mm lateral of the outline axis)
    # flatter (boxier) last section across the instep: throat edges sit ~3 mm below the ridge
    nexp = N_EXP + 0.65 * smoothstep(0.26, 0.40, u) * (1 - smoothstep(0.66, 0.84, u))
    return dict(u=u, al=al, am=am, zfl=zfl, zfm=zfm, ztop=ztop, yt=yt, zc=zc, nexp=nexp)


def arch_point(p, i, side, phi):
    """point on the last cross-section at station index i, side 'lat'/'med', angle phi in [0,pi/2]"""
    a = p['al'][i] if side == 'lat' else p['am'][i]
    zf = p['zfl'][i] if side == 'lat' else p['zfm'][i]
    yt = p['yt'][i]
    yedge = -a if side == 'lat' else a
    H = p['ztop'][i] - zf
    ne = p['nexp'][i] if 'nexp' in p else N_EXP
    c = np.cos(phi) ** (2 / ne)
    s = np.sin(phi) ** (2 / ne)
    bulge = 0.035 * np.sin(2 * phi) * (1 - 0.6 * np.sin(phi))
    y = yt + (yedge - yt) * c * (1 + bulge)
    z = zf + H * s
    return y, z


def phi_end(p, i, side, k=0.035):
    """angle where the knit wall ends (collar / throat edge), soft-min of both cuts."""
    phis = np.linspace(0, np.pi / 2, 900)
    y, z = arch_point(p, i, side, phis)
    u = p['u'][i]
    zc = p['zc'][i]
    pc = np.pi / 2
    above = np.nonzero(z > zc)[0]
    if len(above):
        j = max(above[0] - 1, 0)
        # linear interpolation of the crossing
        z0, z1 = z[j], z[min(j + 1, len(z) - 1)]
        t = 0.0 if z1 == z0 else np.clip((zc - z0) / (z1 - z0), 0, 1)
        pc = phis[j] + t * (phis[1] - phis[0])
    g = float(throat_f(u)) if 0.355 <= u <= 0.716 else 0.0
    pg = np.pi / 2
    if g > 1e-6:
        dy = np.abs(y - p['yt'][i])
        inside = np.nonzero(dy < g)[0]
        if len(inside):
            pg = phis[max(inside[0] - 1, 0)]
    m = min(pc, pg)
    if u > 0.46 or m >= np.pi / 2 - 1e-9:
        return m
    # soft minimum -> rounded corner where the collar meets the throat
    sm = -k * np.log(np.exp(-(pc - m) / k) + np.exp(-(pg - m) / k)) + m
    return min(sm, np.pi / 2)


def upper_surface(nu=300, nt=110):
    """returns grid (nu, 2*nt-1, 3): lateral edge -> lateral featherline (bottom) ... medial edge.
    Ring order: tau from lateral top edge (j=0) down to lateral featherline (j=nt-1),
    then medial featherline up to medial top edge. (bottom is open: hidden in midsole)"""
    s = np.linspace(0, 1, nu)
    u = UB + (UF - UB) * (0.5 * (0.5 - 0.5 * np.cos(np.pi * s)) + 0.5 * s)
    p = upper_params(u)
    pe = {'lat': np.zeros(nu), 'med': np.zeros(nu)}
    for side in ('lat', 'med'):
        for i in range(nu):
            pe[side][i] = phi_end(p, i, side)
        # smooth the edge angle along u (rounded corners)
        raw = pe[side].copy()
        sm = gaussian_filter1d(raw, 2.0, mode='nearest')
        closed = (raw >= np.pi / 2 - 1e-9) & (u > 0.5)
        pe[side] = np.where(closed, np.pi / 2, np.minimum(sm, np.pi / 2))
    grid = np.zeros((nu, 2 * nt, 3))
    tau = np.zeros((nu, 2 * nt))
    for i in range(nu):
        for side, sl in (('lat', slice(0, nt)), ('med', slice(nt, 2 * nt))):
            phis = np.linspace(0, pe[side][i], 400)
            y, z = arch_point(p, i, side, phis)
            poly = np.stack([y, z], 1)
            rs = _resample(poly, nt)     # featherline -> edge
            if side == 'lat':
                rs = rs[::-1]            # edge -> featherline
            grid[i, sl, 0] = ux(u[i])
            grid[i, sl, 1] = rs[:, 0]
            grid[i, sl, 2] = rs[:, 1]
            tt = np.linspace(0, 1, nt)
            tau[i, sl] = tt[::-1] if side == 'lat' else tt
    return u, p, pe, grid, tau


def heel_back_deform(V, u):
    """lean of the heel counter: back silhouette moves forward going up (photo)."""
    z = V[..., 2]
    h = z / L
    shift = np.interp(h, [0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44],
                      [0.0, 0.002, 0.008, 0.015, 0.018, 0.019, 0.019]) * L
    w = 1 - smoothstep(UB, UB + 0.12, u)
    V[..., 0] += shift * w
    return V


# ----------------------------------------------------------------------------
# v2: perimeter grid with a polar (radial-plane) heel cap -> rounded collar U
# ----------------------------------------------------------------------------
UHC = UB + 0.115          # end of the heel cap (polar region)


def _arch_rz(R, zf, H, phi):
    c = np.cos(phi) ** (2 / N_EXP)
    s = np.sin(phi) ** (2 / N_EXP)
    bulge = 0.035 * np.sin(2 * phi) * (1 - 0.6 * np.sin(phi))
    return R * c * (1 + bulge), zf + H * s


def heel_polar_params(psi):
    """featherline radius, heights and arch for the polar heel cap at angles psi (0 lat, pi/2 back, pi med)."""
    psi = np.atleast_1d(psi)
    pc = upper_params(np.array([UHC]))
    xc = ux(UHC)
    yc = float(pc['yt'][0])
    # featherline outline of the heel cap (from upper_params)
    uu = np.linspace(UB, UHC, 400)
    pp = upper_params(uu)
    lat = np.stack([ux(uu), -pp['al']], 1)
    med = np.stack([ux(uu), pp['am']], 1)
    poly = np.concatenate([lat[::-1], med[1:]], 0)       # lateral(UHC)->back->medial(UHC)
    ang = np.arctan2(-(poly[:, 0] - xc), -(poly[:, 1] - yc))   # psi such that d=(-sin,-cos)
    ang = np.unwrap(ang)
    rad = np.hypot(poly[:, 0] - xc, poly[:, 1] - yc)
    o = np.argsort(ang)
    R = np.interp(psi, ang[o], rad[o])
    lean = 0.0090 - 0.0035 * np.sin(psi) ** 2
    d = np.stack([-np.sin(psi), -np.cos(psi)], 1)
    xf = xc + R * d[:, 0]
    uf = xu(xf)
    pf = upper_params(np.clip(uf, UB, UHC))
    wmed = (1 - np.cos(psi)) / 2
    zf = (1 - wmed) * pf['zfl'] + wmed * pf['zfm']
    xe = xc + (R - lean) * d[:, 0]
    zc = collar_h(np.clip(xu(xe), 0.0, 1.0)) * L - 0.0036
    frac = np.clip(1 - lean / np.maximum(R, 1e-4), 0.05, 0.995)
    fh = (1 - frac ** N_EXP) ** (1 / N_EXP)
    H = (zc - zf) / np.maximum(fh, 0.05)
    return dict(psi=psi, R=R, zf=zf, zc=zc, H=H, d=d, xc=xc, yc=yc, uf=uf, fh=fh)


def upper_surface_v2(nB=230, nA=90, nt=110):
    """perimeter grid (n_perim, nt): each row a wall profile featherline (col 0) -> top edge (col nt-1).
    rows: lateral side toe->UHC, heel cap (polar), medial side UHC->toe."""
    s = np.linspace(0, 1, nB)
    u = UHC + (UF - UHC) * (0.5 * (0.5 - 0.5 * np.cos(np.pi * s)) + 0.5 * s)
    p = upper_params(u)
    pe = {'lat': np.zeros(nB), 'med': np.zeros(nB)}
    for side in ('lat', 'med'):
        for i in range(nB):
            pe[side][i] = phi_end(p, i, side)
        raw = pe[side].copy()
        sm = gaussian_filter1d(raw, 2.0, mode='nearest')
        closed = (raw >= np.pi / 2 - 1e-9) & (u > 0.5)
        pe[side] = np.where(closed, np.pi / 2, np.minimum(sm, np.pi / 2))
    rows, rinfo = [], []
    tt = np.linspace(0, 1, nt)

    def B_row(i, side):
        phis = np.linspace(0, pe[side][i], 500)
        y, z = arch_point(p, i, side, phis)
        rs = _resample(np.stack([y, z], 1), nt)
        return np.stack([np.full(nt, ux(u[i])), rs[:, 0], rs[:, 1]], 1)

    for i in range(nB - 1, -1, -1):
        rows.append(B_row(i, 'lat')); rinfo.append(('lat', u[i], None))
    psi = np.linspace(0, np.pi, nA)[1:-1]
    hp = heel_polar_params(psi)
    for k in range(len(psi)):
        R, zf, H = hp['R'][k], hp['zf'][k], hp['H'][k]
        # phi where z reaches the collar edge
        fe = np.clip((hp['zc'][k] - zf) / H, 0, 1)
        pend = np.arcsin(fe ** (N_EXP / 2))
        phis = np.linspace(0, pend, 500)
        r, z = _arch_rz(R, zf, H, phis)
        rs = _resample(np.stack([r, z], 1), nt)
        x = hp['xc'] + rs[:, 0] * hp['d'][k, 0]
        y = hp['yc'] + rs[:, 0] * hp['d'][k, 1]
        rows.append(np.stack([x, y, rs[:, 1]], 1)); rinfo.append(('heel', float(hp['uf'][k]), float(psi[k])))
    for i in range(nB):
        rows.append(B_row(i, 'med')); rinfo.append(('med', u[i], None))
    grid = np.stack(rows, 0)
    return dict(grid=grid, rinfo=rinfo, u=u, p=p, pe=pe, nB=nB, nA=len(psi), nt=nt, hp=hp)
