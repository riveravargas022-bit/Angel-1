"""Assemble the Olympikus Lumi digital twin in Blender (geometry + materials) -> lumi.blend"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import bl_util as bu
import shoe_geom as g
import shoe_parts as sp
import materials as mt

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
CLAY = 'clay' in ARGS
OUT = os.path.join(HERE, 'out', 'lumi_clay.blend' if CLAY else 'lumi.blend')
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
MM = 1e-3

bu.reset()
sc = bpy.context.scene
col = bpy.data.collections.new('Olympikus_Lumi')
sc.collection.children.link(col)


def obj(name, V, F, smooth=True):
    return bu.mesh_from(name, V, F, smooth, coll=col)


def set_uv(ob, uv_per_loop_fn=None, uv_vert=None, name='UVMap'):
    me = ob.data
    uvl = me.uv_layers.new(name=name)
    if uv_vert is not None:
        li = np.zeros(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', li)
        uvl.data.foreach_set('uv', np.asarray(uv_vert, np.float32)[li].ravel())
    return uvl


# ---------------------------------------------------------------- midsole
V, F, uu = g.build_midsole()
mid = obj('Midsole', V, F)
bu.add_float_attr(mid, 'u_len', uu)

# ---------------------------------------------------------------- upper
up = sp.Upper()
upo = obj('Upper_Knit', up.V, up.F)
if up.flipped:
    # make faces point outward
    bm = bmesh.new(); bm.from_mesh(upo.data)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(upo.data); bm.free()
bu.add_float_attr(upo, 'u_len', up.vu)
bu.add_float_attr(upo, 'tau', up.vtau)
_uvl = upo.data.uv_layers.new(name='UVMap')
assert len(upo.data.loops) == len(up.loop_uv), (len(upo.data.loops), len(up.loop_uv))
_uvl.data.foreach_set('uv', up.loop_uv.astype(np.float32).ravel())
sol = upo.modifiers.new('Thickness', 'SOLIDIFY')
sol.thickness = 2.2 * MM
sol.offset = -1.0
sol.use_rim = True
sol.material_offset = 1          # inner shell -> lining material
sol.material_offset_rim = 1
sol.use_even_offset = True

# ---------------------------------------------------------------- collar padding
Vc, Fc, cinfo = sp.collar_mesh(up)
collar = obj('Collar_Padding', Vc, Fc)
bu.merge_by_distance(collar, 1e-7)

# ---------------------------------------------------------------- tongue
Vt, Ft, At, tg = sp.tongue_mesh(up)
tongue = obj('Tongue', Vt, Ft)
for k, a in At.items():
    bu.add_float_attr(tongue, k, a)
set_uv(tongue, uv_vert=np.stack([0.5 + At['ty'] / 0.060, 1.0 - At['tarc'] / 0.110], 1))

# ---------------------------------------------------------------- eyestay (TPU overlay)
Ve, Fe, es = sp.eyestay_mesh(up)
eyestay = obj('Eyestay_TPU', Ve, Fe)
set_uv(eyestay, uv_vert=np.stack([es['sa'], es['ta']], 1))
bu.add_float_attr(eyestay, 'es_s', es['sa'])
bu.add_float_attr(eyestay, 'es_t', es['ta'])
es_sol = eyestay.modifiers.new('Thickness', 'SOLIDIFY')
es_sol.thickness = 0.9 * MM
es_sol.offset = 1.0
es_sol.use_even_offset = False
es_sub = eyestay.modifiers.new('Soft', 'SUBSURF')
es_sub.levels = 1
es_sub.render_levels = 2

# eyelets: punched-hole look = dark elliptical floor + soft TPU rim (no boolean artefacts)
eyes = sp.eyelet_points(up, es)
hole_V, hole_F, rim_V, rim_F = [], [], [], []
for e in eyes:
    n = sp.norm(np.asarray(e['N'])); T_ = sp.norm(np.asarray(e['T']))
    T_ = sp.norm(T_ - (T_ @ n) * n); B_ = np.cross(n, T_)
    P0 = np.asarray(e['P']) + n * (0.45 + 0.9 + 0.04) * MM      # top of TPU
    a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rx, ry = 2.15 * MM, 1.65 * MM                                  # along edge / across
    base = len(hole_V)
    hole_V.append(P0)
    for k in range(32):
        hole_V.append(P0 + np.cos(a[k]) * rx * T_ + np.sin(a[k]) * ry * B_)
    for k in range(32):
        hole_F.append((base, base + 1 + k, base + 1 + (k + 1) % 32))
    # rim: torus-like ring slightly proud
    rb = len(rim_V)
    for k in range(32):
        for j, (dr, dz) in enumerate(((-0.15, -0.05), (0.25, 0.18), (0.75, 0.22), (1.2, 0.0))):
            rim_V.append(P0 + np.cos(a[k]) * (rx + dr * MM) * T_ + np.sin(a[k]) * (ry + dr * MM) * B_ + n * dz * MM)
    for k in range(32):
        k2 = (k + 1) % 32
        for j in range(3):
            rim_F.append((rb + k * 4 + j, rb + k * 4 + j + 1, rb + k2 * 4 + j + 1, rb + k2 * 4 + j))
holes = obj('Eyelet_Holes', np.array(hole_V), hole_F, smooth=False)
rims = obj('Eyelet_Rims', np.array(rim_V), rim_F)

# ---------------------------------------------------------------- throat binding
Vp_, Fp_ = sp.piping_mesh(up)
piping = obj('Throat_Binding', Vp_, Fp_)

# ---------------------------------------------------------------- laces
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
bvh_tops = [BVHTree.FromObject(o, deps) for o in (tongue, eyestay, upo)]


def surf_z(xa, ya):
    out = np.zeros(len(xa))
    for k, (x_, y_) in enumerate(zip(xa, ya)):
        best = -1.0
        for bvh_ in bvh_tops:
            hit = bvh_.ray_cast(Vector((x_, y_, 0.4)), Vector((0, 0, -1)))
            if hit[0] is not None and hit[0].z > best:
                best = hit[0].z
        out[k] = best if best > 0 else 0.05
    return out


Vl, Fl, UVl = sp.lace_mesh(eyes, surf_z)
laces = obj('Laces', Vl, Fl)
set_uv(laces, uv_vert=UVl)

# ---------------------------------------------------------------- heel tab
Vh, Fh, hside, _ = sp.heel_tab_mesh(up, cinfo)
heel_tab = obj('Heel_Pull_Tab', Vh, Fh)
bu.add_float_attr(heel_tab, 'tab_side', hside)

# ---------------------------------------------------------------- insole
Vi, Fi = sp.insole_mesh(up)
insole = obj('Insole', Vi, Fi)
ins_sol = insole.modifiers.new('Thickness', 'SOLIDIFY')
ins_sol.thickness = 3.0 * MM
ins_sol.offset = -1

# ---------------------------------------------------------------- side badge (raised logo)
from PIL import Image
info = json.load(open(os.path.join(HERE, 'tex', 'badge_info.json')))
hmap = np.asarray(Image.open(os.path.join(HERE, 'tex', 'badge_height.png'))).astype(np.float32) / 65535.0
step = 4                                     # sample every 4 px of the 8x mask (~0.032 mm)
hs = hmap[::step, ::step]
H, W = hs.shape
side1 = g.S['side1']
Lp = side1['x1'] - side1['x0']
px_full = 1.0 / info['px_per_crop'] * step   # photo px per sample
cols = info['crop_x0'] + np.arange(W) * px_full
rows = info['crop_y0'] + np.arange(H) * px_full
xs = g.X0 + (cols - side1['x0']) / Lp * g.L
zs = (side1['ground'] - rows) / Lp * g.L
# ray-cast onto evaluated upper (with thickness modifier)
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(upo, deps)
Hmm = 1.05 * MM
keep = hs > 0.004
Vb = np.zeros((H, W, 3))
okm = np.zeros((H, W), bool)
for r in range(H):
    for c in range(W):
        if not keep[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any():
            continue
        loc, nrm, idx, dist = bvh.ray_cast(Vector((xs[c], -0.2, zs[r])), Vector((0, 1, 0)))
        if loc is None:
            continue
        n = np.array(nrm)
        if n[1] > 0:
            n = -n
        Vb[r, c] = np.array(loc) + n * (hs[r, c] * Hmm - 0.15 * MM)
        okm[r, c] = True
idx = -np.ones((H, W), np.int64)
verts = []
uvs = []
for r in range(H):
    for c in range(W):
        if okm[r, c]:
            idx[r, c] = len(verts)
            verts.append(Vb[r, c])
            uvs.append(((c * step + 0.5) / hmap.shape[1], 1 - (r * step + 0.5) / hmap.shape[0]))
faces = []
for r in range(H - 1):
    for c in range(W - 1):
        q = (idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c])
        if min(q) < 0:
            continue
        if max(hs[r, c], hs[r, c + 1], hs[r + 1, c + 1], hs[r + 1, c]) < 0.004:
            continue
        faces.append(q[::-1])
badge = obj('Logo_Badge', np.array(verts), faces)
set_uv(badge, uv_vert=np.array(uvs))

# ---------------------------------------------------------------- materials
M = mt.build_all(CLAY)
mid.data.materials.append(M['midsole'])
upo.data.materials.append(M['knit'])
upo.data.materials.append(M['lining'])
collar.data.materials.append(M['lining'])
tongue.data.materials.append(M['tongue'])
eyestay.data.materials.append(M['tpu'])
holes.data.materials.append(M['hole'])
rims.data.materials.append(M['tpu_plain'])
piping.data.materials.append(M['lining'])
laces.data.materials.append(M['lace'])
heel_tab.data.materials.append(M['heeltab'])
insole.data.materials.append(M['insole'])
badge.data.materials.append(M['badge'])

# EVASENSE decal on the medial midsole: small projected plane object
ev_u0, ev_u1, ev_h = 0.262, 0.334, 0.1215
x0, x1 = g.ux(ev_u0), g.ux(ev_u1)
zc = ev_h * g.L
nx, nz = 120, 16
Wd, Hd = x1 - x0, 3.6 * MM * (x1 - x0) / (16.5 * MM)
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
bvh_mid = BVHTree.FromObject(mid, deps)
dv, dfc, duv = [], [], []
for j in range(nz):
    for i in range(nx):
        x = x1 - Wd * i / (nx - 1)          # text reads toward the heel on the medial side
        z = zc + Hd * (0.5 - j / (nz - 1))
        loc, nrm, _, _ = bvh_mid.ray_cast(Vector((x, 0.2, z)), Vector((0, -1, 0)))
        if loc is None:
            loc, nrm = Vector((x, 0.04, z)), Vector((0, 1, 0))
        dv.append(np.array(loc) + np.array(nrm) * 0.06 * MM)
        duv.append((i / (nx - 1), 1 - j / (nz - 1)))
for j in range(nz - 1):
    for i in range(nx - 1):
        a = j * nx + i
        dfc.append((a, a + nx, a + nx + 1, a + 1))
deca = obj('EVASENSE_Print', np.array(dv), dfc)
set_uv(deca, uv_vert=np.array(duv))
deca.data.materials.append(M['evasense'])

# parent everything to an empty for easy transforms
root = bpy.data.objects.new('Olympikus_Lumi_Root', None)
col.objects.link(root)
for o in list(col.objects):
    if o is not root:
        o.parent = root

json.dump({'eyes': [{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in e.items()} for e in eyes]},
          open(os.path.join(HERE, 'out', 'eyes.json'), 'w'))
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print('saved', OUT)
