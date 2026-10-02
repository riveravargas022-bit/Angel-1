"""PBR materials for the Lumi digital twin (Cycles)."""
import os, json
import numpy as np
import bpy
import bl_util as bu
import shoe_geom as g

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = lambda n: os.path.join(HERE, 'tex', n)
L = g.L


def lin(hx):
    return (*bu.hex_lin(hx), 1.0)


class NT:
    """tiny node-tree builder"""

    def __init__(self, mat):
        self.m = mat
        self.t = mat.node_tree
        self.n = self.t.nodes
        self.l = self.t.links
        self.x = -1600

    def node(self, typ, **kw):
        nd = self.n.new(typ)
        nd.location = (self.x, 0)
        self.x += 40
        for k, v in kw.items():
            if k.startswith('in_'):
                key = k[3:]
                key = int(key) if key.isdigit() else key.replace('__', ' ')
                nd.inputs[key].default_value = v
            else:
                setattr(nd, k, v)
        return nd

    def link(self, a, b):
        self.l.new(a, b)

    def math(self, op, a=None, b=None, clamp=False, va=0.0, vb=0.0):
        nd = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        nd.inputs[0].default_value = va
        nd.inputs[1].default_value = vb
        if a is not None:
            self.link(a, nd.inputs[0])
        if b is not None:
            self.link(b, nd.inputs[1])
        return nd.outputs[0]

    def img(self, path, colorspace='sRGB', ext='CLIP', interp='Cubic', proj='FLAT'):
        im = bpy.data.images.load(path, check_existing=True)
        im.colorspace_settings.name = colorspace
        nd = self.node('ShaderNodeTexImage', image=im, extension=ext, interpolation=interp, projection=proj)
        return nd


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes['Principled BSDF'], NT(m)


def set_bsdf(b, **kw):
    for k, v in kw.items():
        b.inputs[k.replace('_', ' ')].default_value = v


# ---------------------------------------------------------------------------

def knit_material():
    m, b, nt = new_mat('Knit_Jacquard_Lilac')
    meta = json.load(open(TEX('knit_meta.json')))
    tc = nt.node('ShaderNodeTexCoord')
    sep = nt.node('ShaderNodeSeparateXYZ')
    nt.link(tc.outputs['Object'], sep.inputs[0])
    x, y, z = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    sepn = nt.node('ShaderNodeSeparateXYZ')
    nt.link(tc.outputs['Normal'], sepn.inputs[0])          # object-space normal (pose independent)
    nx, ny, nz = sepn.outputs[0], sepn.outputs[1], sepn.outputs[2]

    def side_uv(mt_, im_w, im_h):
        # U = (x0 + (x-X0)/L*Lp - cx0)/(cx1-cx0) ; flipped for medial
        cx0, cy0, cx1, cy1 = mt_['cx0'], mt_['cy0'], mt_['cx1'], mt_['cy1']
        Lp, x0, gr = mt_['Lp'], mt_['x0'], mt_['ground']
        if not mt_['flip']:
            a = Lp / L / (cx1 - cx0)
            bconst = (x0 - cx0 - g.X0 * Lp / L) / (cx1 - cx0)
        else:
            a = -Lp / L / (cx1 - cx0)
            bconst = (x0 + Lp - cx0 + g.X0 * Lp / L) / (cx1 - cx0)
        U = nt.math('MULTIPLY_ADD', x, None, va=0, vb=a)
        U.node.inputs[2].default_value = bconst
        # V = 1 - (gr - z*Lp/L - cy0)/(cy1-cy0)
        az = (Lp / L) / (cy1 - cy0)
        bz = 1 - (gr - cy0) / (cy1 - cy0)
        V = nt.math('MULTIPLY_ADD', z, None, vb=az)
        V.node.inputs[2].default_value = bz
        comb = nt.node('ShaderNodeCombineXYZ')
        nt.link(U, comb.inputs[0]); nt.link(V, comb.inputs[1])
        return comb.outputs[0]

    uv_lat = side_uv(meta['lat'], 0, 0)
    uv_med = side_uv(meta['med'], 0, 0)
    # top projection with perspective (camera ~0.51 m above ground)
    tp = meta['top']
    Lt = tp['row_toe'] - tp['row_heel']
    D = 1.94 * L
    zref = 0.10 * L
    k = nt.math('DIVIDE', None, nt.math('SUBTRACT', None, z, va=D), va=(D - zref))   # k=(D-zref)/(D-z)
    # flat mapping: row = row_heel + u*Lt ; col = axis(u) + y/L/WS*Lt
    WS = g.WIDTH_SCALE
    ah, at_ = tp['axis_c_heel'], tp['axis_c_toe']
    slope = (at_ - ah) / (0.985 - 0.08)
    # u = (x - X0)/L
    u = nt.math('MULTIPLY_ADD', x, None, vb=1 / L)
    u.node.inputs[2].default_value = -g.X0 / L
    row = nt.math('MULTIPLY_ADD', u, None, vb=Lt)
    row.node.inputs[2].default_value = tp['row_heel']
    axis = nt.math('MULTIPLY_ADD', u, None, vb=slope)
    axis.node.inputs[2].default_value = ah - slope * 0.08
    col = nt.math('ADD', axis, nt.math('MULTIPLY', y, None, vb=Lt / L / WS))
    rc, cc = tp['H'] / 2, tp['W'] / 2
    rowp = nt.math('ADD', nt.math('MULTIPLY', nt.math('SUBTRACT', row, None, vb=rc), k), None, vb=rc)
    colp = nt.math('ADD', nt.math('MULTIPLY', nt.math('SUBTRACT', col, None, vb=cc), k), None, vb=cc)
    Ut = nt.math('MULTIPLY_ADD', colp, None, vb=1 / (tp['cx1'] - tp['cx0']))
    Ut.node.inputs[2].default_value = -tp['cx0'] / (tp['cx1'] - tp['cx0'])
    Vt = nt.math('MULTIPLY_ADD', rowp, None, vb=-1 / (tp['cy1'] - tp['cy0']))
    Vt.node.inputs[2].default_value = 1 + tp['cy0'] / (tp['cy1'] - tp['cy0'])
    combt = nt.node('ShaderNodeCombineXYZ')
    nt.link(Ut, combt.inputs[0]); nt.link(Vt, combt.inputs[1])
    uv_top = combt.outputs[0]

    # textures
    def tex3(base, uv):
        c = nt.img(TEX(f'knit_{base}.png'))
        h = nt.img(TEX(f'knit_{base}_h.png'), 'Non-Color')
        mk = nt.img(TEX(f'knit_{base}_mask.png'), 'Non-Color')
        for nd in (c, h, mk):
            nt.link(uv, nd.inputs['Vector'])
        return c.outputs['Color'], h.outputs['Color'], mk.outputs['Color']

    c_lat, h_lat, m_lat = tex3('lat', uv_lat)
    c_med, h_med, m_med = tex3('med', uv_med)
    c_top, h_top, m_top = tex3('top', uv_top)
    # tile (box mapped) for unseen areas
    mp = nt.node('ShaderNodeMapping')
    nt.link(tc.outputs['Object'], mp.inputs['Vector'])
    tile_w = 768 / 4207 * L
    mp.inputs['Scale'].default_value = (1 / tile_w, 1 / tile_w, 1 / (512 / 4207 * L) * (512 / 768) * 1.0)
    ct = nt.img(TEX('knit_tile.png'), ext='REPEAT', proj='BOX')
    ct.projection_blend = 0.3
    ht = nt.img(TEX('knit_tile_h.png'), 'Non-Color', ext='REPEAT', proj='BOX')
    ht.projection_blend = 0.3
    nt.link(mp.outputs[0], ct.inputs['Vector']); nt.link(mp.outputs[0], ht.inputs['Vector'])

    P = 3.0

    def w_of(comp, sign, mask):
        w = nt.math('POWER', nt.math('MAXIMUM', nt.math('MULTIPLY', comp, None, vb=sign), None, vb=0.0), None, vb=P)
        bwm = nt.node('ShaderNodeRGBToBW'); nt.link(mask, bwm.inputs[0])
        return nt.math('MULTIPLY', w, bwm.outputs[0])

    wl = w_of(ny, -1.0, m_lat)
    wm = w_of(ny, 1.0, m_med)
    wt = w_of(nz, 1.0, m_top)
    wt = nt.math('MULTIPLY', wt, None, vb=0.8)
    wtile = nt.math('ADD', nt.math('MULTIPLY', nt.math('POWER', nt.math('ABSOLUTE', nx), None, vb=2.0), None, vb=0.25), None, vb=0.004)
    tot = nt.math('ADD', nt.math('ADD', wl, wm), nt.math('ADD', wt, wtile))

    def mixsum(items):
        acc = None
        for col_, w_ in items:
            wn = nt.math('DIVIDE', w_, tot)
            mv = nt.node('ShaderNodeVectorMath', operation='SCALE')
            nt.link(col_, mv.inputs[0]); nt.link(wn, mv.inputs['Scale'])
            if acc is None:
                acc = mv.outputs[0]
            else:
                ad = nt.node('ShaderNodeVectorMath', operation='ADD')
                nt.link(acc, ad.inputs[0]); nt.link(mv.outputs[0], ad.inputs[1])
                acc = ad.outputs[0]
        return acc

    col_ = mixsum([(c_lat, wl), (c_med, wm), (c_top, wt), (ct.outputs['Color'], wtile)])
    hgt = mixsum([(h_lat, wl), (h_med, wm), (h_top, wt), (ht.outputs['Color'], wtile)])
    # tone: photos de-lit to ~205/255 ; keep slightly brighter for albedo
    cc_ = nt.node('ShaderNodeHueSaturation')
    cc_.inputs['Saturation'].default_value = 1.18
    cc_.inputs['Value'].default_value = 1.07
    nt.link(col_, cc_.inputs['Color'])
    nt.link(cc_.outputs[0], b.inputs['Base Color'])
    # micro rib (horizontal courses, 0.55 mm pitch) for macro shots
    wv = nt.node('ShaderNodeTexWave', wave_type='BANDS', bands_direction='Z', wave_profile='SIN')
    wv.inputs['Scale'].default_value = 1.0
    mpw = nt.node('ShaderNodeMapping')
    nt.link(tc.outputs['Object'], mpw.inputs['Vector'])
    mpw.inputs['Scale'].default_value = (1 / 0.0011, 1 / 0.0011, 1 / 0.0011)
    nt.link(mpw.outputs[0], wv.inputs['Vector'])
    wv.inputs['Distortion'].default_value = 0.6
    wv.inputs['Detail'].default_value = 2.0
    hb = nt.node('ShaderNodeRGBToBW'); nt.link(hgt, hb.inputs[0])
    hsum = nt.math('ADD', hb.outputs[0], nt.math('MULTIPLY', wv.outputs['Fac'], None, vb=0.25))
    bump = nt.node('ShaderNodeBump', invert=False)
    bump.inputs['Strength'].default_value = 0.55
    bump.inputs['Distance'].default_value = 0.00045
    nt.link(hsum, bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    set_bsdf(b, Roughness=0.78)
    b.inputs['Specular IOR Level'].default_value = 0.35
    b.inputs['Sheen Weight'].default_value = 0.55
    b.inputs['Sheen Roughness'].default_value = 0.45
    b.inputs['Sheen Tint'].default_value = lin('#E6E2F5')
    return m


def lining_material():
    m, b, nt = new_mat('Lining_Collar_Periwinkle')
    tc = nt.node('ShaderNodeTexCoord')
    mp = nt.node('ShaderNodeMapping')
    nt.link(tc.outputs['Object'], mp.inputs['Vector'])
    mp.inputs['Rotation'].default_value = (0.5, 0.3, 0.7)
    mp.inputs['Scale'].default_value = (1 / 0.0007,) * 3
    wv = nt.node('ShaderNodeTexWave', wave_type='BANDS', bands_direction='X', wave_profile='SIN')
    nt.link(mp.outputs[0], wv.inputs['Vector'])
    wv.inputs['Distortion'].default_value = 0.3
    bump = nt.node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.25
    bump.inputs['Distance'].default_value = 0.0002
    nt.link(wv.outputs['Fac'], bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    set_bsdf(b, Roughness=0.72)
    b.inputs['Base Color'].default_value = lin('#A2A9C8')
    b.inputs['Sheen Weight'].default_value = 0.7
    b.inputs['Sheen Roughness'].default_value = 0.35
    b.inputs['Sheen Tint'].default_value = lin('#C9D0EE')
    b.inputs['Specular IOR Level'].default_value = 0.3
    return m


def midsole_material():
    m, b, nt = new_mat('Midsole_EVA_OffWhite')
    tc = nt.node('ShaderNodeTexCoord')
    sep = nt.node('ShaderNodeSeparateXYZ'); nt.link(tc.outputs['Object'], sep.inputs[0])
    sepn = nt.node('ShaderNodeSeparateXYZ'); nt.link(tc.outputs['Normal'], sepn.inputs[0])
    # micro foam grain
    nz_ = nt.node('ShaderNodeTexNoise')
    nz_.inputs['Scale'].default_value = 5200.0
    nz_.inputs['Detail'].default_value = 3.0
    nt.link(tc.outputs['Object'], nz_.inputs['Vector'])
    fit = json.load(open(TEX('outsole_fit.json')))
    Ux = nt.math('MULTIPLY', sep.outputs[0], None, vb=fit['aux'])
    U = nt.math('MULTIPLY_ADD', sep.outputs[1], None, vb=fit['auy'])
    U.node.inputs[2].default_value = fit['bu']
    U = nt.math('ADD', U, Ux)
    V = nt.math('MULTIPLY_ADD', sep.outputs[0], None, vb=fit['av'])
    V.node.inputs[2].default_value = fit['bv']
    comb = nt.node('ShaderNodeCombineXYZ'); nt.link(U, comb.inputs[0]); nt.link(V, comb.inputs[1])
    th = nt.img(TEX('outsole_h.png'), 'Non-Color')
    nt.link(comb.outputs[0], th.inputs['Vector'])
    downw = nt.math('MAXIMUM', nt.math('MULTIPLY', sepn.outputs[2], None, vb=-1.0), None, vb=0.0)
    downw = nt.math('POWER', downw, None, vb=4.0)
    tread = nt.math('MULTIPLY', th.outputs['Color'], downw)
    hsum = nt.math('ADD', nt.math('MULTIPLY', nz_.outputs['Fac'], None, vb=0.04), tread)
    bump = nt.node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 1.0
    bump.inputs['Distance'].default_value = 0.0016
    nt.link(hsum, bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    b.inputs['Base Color'].default_value = lin('#EBE5DA')
    set_bsdf(b, Roughness=0.58)
    b.inputs['Specular IOR Level'].default_value = 0.42
    b.inputs['Subsurface Weight'].default_value = 0.05
    b.inputs['Subsurface Radius'].default_value = (1.0, 0.85, 0.7)
    b.inputs['Subsurface Scale'].default_value = 0.0015
    return m


def tpu_material(es_tex=None):
    m, b, nt = new_mat('Eyestay_TPU_Lilac')
    base = lin('#C4C1E3')
    if es_tex:
        uvn = nt.node('ShaderNodeUVMap')
        im = nt.img(TEX('eyestay.png'), 'Non-Color', ext='EXTEND')
        nt.link(uvn.outputs[0], im.inputs['Vector'])
        mx = nt.node('ShaderNodeMix', data_type='RGBA')
        mx.inputs[6].default_value = base
        mx.inputs[7].default_value = lin('#86DCC6')
        sepc = nt.node('ShaderNodeSeparateColor')
        nt.link(im.outputs['Color'], sepc.inputs[0])
        nt.link(sepc.outputs[0], mx.inputs['Factor'])
        nt.link(mx.outputs[2], b.inputs['Base Color'])
        bump = nt.node('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 1.0
        bump.inputs['Distance'].default_value = 0.0007
        nt.link(sepc.outputs[1], bump.inputs['Height'])
        nt.link(bump.outputs[0], b.inputs['Normal'])
    else:
        b.inputs['Base Color'].default_value = base
    set_bsdf(b, Roughness=0.42)
    b.inputs['Specular IOR Level'].default_value = 0.5
    b.inputs['Coat Weight'].default_value = 0.12
    b.inputs['Coat Roughness'].default_value = 0.35
    return m


def badge_material():
    m, b, nt = new_mat('Logo_Badge_TPU')
    uvn = nt.node('ShaderNodeUVMap')
    im = nt.img(TEX('badge_mint.png'), 'Non-Color', ext='EXTEND')
    nt.link(uvn.outputs[0], im.inputs['Vector'])
    mx = nt.node('ShaderNodeMix', data_type='RGBA')
    mx.inputs[6].default_value = lin('#8B8AB4')
    mx.inputs[7].default_value = lin('#7ED9C3')
    nt.link(im.outputs['Color'], mx.inputs['Factor'])
    nt.link(mx.outputs[2], b.inputs['Base Color'])
    set_bsdf(b, Roughness=0.33)
    b.inputs['Specular IOR Level'].default_value = 0.55
    b.inputs['Coat Weight'].default_value = 0.25
    b.inputs['Coat Roughness'].default_value = 0.25
    return m


def tongue_material(tongue_tex):
    m, b, nt = new_mat('Tongue_Mesh_Mint_Label')
    uvn = nt.node('ShaderNodeUVMap')
    im = nt.img(TEX('tongue.png'), 'sRGB', ext='EXTEND')
    nt.link(uvn.outputs[0], im.inputs['Vector'])
    imh = nt.img(TEX('tongue_h.png'), 'Non-Color', ext='EXTEND')
    nt.link(uvn.outputs[0], imh.inputs['Vector'])
    front = nt.node('ShaderNodeAttribute', attribute_name='tfront')
    # back side of tongue: plain mesh color
    mx = nt.node('ShaderNodeMix', data_type='RGBA')
    mx.inputs[6].default_value = lin('#A9E3D4')
    nt.link(front.outputs['Fac'], mx.inputs['Factor'])
    nt.link(im.outputs['Color'], mx.inputs[7])
    nt.link(mx.outputs[2], b.inputs['Base Color'])
    bump = nt.node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.8
    bump.inputs['Distance'].default_value = 0.0007
    nt.link(imh.outputs['Color'], bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    set_bsdf(b, Roughness=0.7)
    b.inputs['Sheen Weight'].default_value = 0.5
    b.inputs['Sheen Roughness'].default_value = 0.4
    b.inputs['Specular IOR Level'].default_value = 0.35
    return m


def lace_material():
    m, b, nt = new_mat('Lace_Flat_Woven_Lilac')
    uvn = nt.node('ShaderNodeUVMap')
    sep = nt.node('ShaderNodeSeparateXYZ'); nt.link(uvn.outputs[0], sep.inputs[0])
    # woven ribs across the lace: phase shifts with V (herringbone-ish twill)
    ph = nt.math('ADD', nt.math('MULTIPLY', sep.outputs[0], None, vb=4.4 * 6.2832),
                 nt.math('MULTIPLY', nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', nt.math('MULTIPLY', sep.outputs[1], None, vb=2.0)), None, vb=0.5)), None, vb=9.0))
    rib = nt.math('MULTIPLY', nt.math('ADD', nt.math('SINE', ph), None, vb=1.0), None, vb=0.5)
    # fine yarn grain
    nz_ = nt.node('ShaderNodeTexNoise'); nz_.inputs['Scale'].default_value = 40.0
    nt.link(uvn.outputs[0], nz_.inputs['Vector'])
    h = nt.math('ADD', rib, nt.math('MULTIPLY', nz_.outputs['Fac'], None, vb=0.35))
    bump = nt.node('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.7
    bump.inputs['Distance'].default_value = 0.0004
    nt.link(h, bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    # colour with slight yarn variation
    cr = nt.node('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].color = lin('#9C9BBA')
    cr.color_ramp.elements[1].color = lin('#B9B8D2')
    nt.link(h, cr.inputs[0])
    nt.link(cr.outputs[0], b.inputs['Base Color'])
    set_bsdf(b, Roughness=0.74)
    b.inputs['Sheen Weight'].default_value = 0.6
    b.inputs['Sheen Roughness'].default_value = 0.35
    b.inputs['Specular IOR Level'].default_value = 0.3
    return m


def heeltab_material():
    m, b, nt = new_mat('Heel_Tab_Webbing')
    at = nt.node('ShaderNodeAttribute', attribute_name='tab_side')
    mx = nt.node('ShaderNodeMix', data_type='RGBA')
    mx.inputs[6].default_value = lin('#8F6DD9')     # inner face purple
    mx.inputs[7].default_value = lin('#CFCBE2')     # outer face light lilac
    nt.link(at.outputs['Fac'], mx.inputs['Factor'])
    nt.link(mx.outputs[2], b.inputs['Base Color'])
    tc = nt.node('ShaderNodeTexCoord')
    wv = nt.node('ShaderNodeTexWave', wave_type='BANDS', bands_direction='Z')
    wv.inputs['Scale'].default_value = 1 / 0.0007
    nt.link(tc.outputs['Object'], wv.inputs['Vector'])
    bump = nt.node('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.3
    bump.inputs['Distance'].default_value = 0.0002
    nt.link(wv.outputs['Fac'], bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    set_bsdf(b, Roughness=0.6)
    b.inputs['Sheen Weight'].default_value = 0.5
    return m


def insole_material():
    m, b, nt = new_mat('Insole_Print')
    info = json.load(open(TEX('insole_info.json')))
    tc = nt.node('ShaderNodeTexCoord')
    sep = nt.node('ShaderNodeSeparateXYZ'); nt.link(tc.outputs['Object'], sep.inputs[0])
    x_start = g.ux(info['start_u'])
    U = nt.math('MULTIPLY_ADD', sep.outputs[0], None, vb=1 / 0.240)
    U.node.inputs[2].default_value = -x_start / 0.240
    V = nt.math('MULTIPLY_ADD', sep.outputs[1], None, vb=1 / 0.100)
    V.node.inputs[2].default_value = 0.5
    comb = nt.node('ShaderNodeCombineXYZ'); nt.link(U, comb.inputs[0]); nt.link(V, comb.inputs[1])
    im = nt.img(TEX('insole_print.png'), 'sRGB', ext='EXTEND')
    nt.link(comb.outputs[0], im.inputs['Vector'])
    mx = nt.node('ShaderNodeMix', data_type='RGBA')
    mx.inputs[6].default_value = lin('#1C1C1E')
    mx.inputs[7].default_value = lin('#EFEFEC')
    nt.link(im.outputs['Color'], mx.inputs['Factor'])
    nt.link(mx.outputs[2], b.inputs['Base Color'])
    nz_ = nt.node('ShaderNodeTexNoise'); nz_.inputs['Scale'].default_value = 3000
    nt.link(tc.outputs['Object'], nz_.inputs['Vector'])
    bump = nt.node('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.15
    nt.link(nz_.outputs['Fac'], bump.inputs['Height'])
    nt.link(bump.outputs[0], b.inputs['Normal'])
    set_bsdf(b, Roughness=0.85)
    b.inputs['Sheen Weight'].default_value = 0.3
    return m


def evasense_material():
    m, b, nt = new_mat('EVASENSE_Print')
    uvn = nt.node('ShaderNodeUVMap')
    im = nt.img(TEX('evasense_alpha.png'), 'Non-Color', ext='CLIP')
    nt.link(uvn.outputs[0], im.inputs['Vector'])
    b.inputs['Base Color'].default_value = lin('#1E1E20')
    set_bsdf(b, Roughness=0.5)
    nt.link(im.outputs['Color'], b.inputs['Alpha'])
    return m


def hole_material():
    m, b, nt = new_mat('Eyelet_Hole_Dark')
    b.inputs['Base Color'].default_value = lin('#25243A')
    set_bsdf(b, Roughness=0.9)
    return m


def clay(name, hx, rough=0.6):
    return bu.principled(name, bu.hex_lin(hx), rough)


def build_all(clay_mode=False, tongue_tex=None, es_tex=None):
    if clay_mode:
        return dict(midsole=clay('c_mid', '#EDE8DF'), knit=clay('c_knit', '#D9D7EC'), lining=clay('c_lin', '#8F97BC'),
                    tongue=clay('c_tg', '#B9B7CB'), tpu=clay('c_tpu', '#C7C5E0', 0.4), hole=clay('c_hole', '#25243A'),
                    lace=clay('c_lace', '#A9A7CB'), heeltab=clay('c_tab', '#8F6DD9'), insole=clay('c_ins', '#EFEFEC'), tpu_plain=clay('c_tpu2', '#C7C5E0', 0.4),
                    badge=clay('c_badge', '#8E8CC6', 0.35), evasense=evasense_material())
    return dict(midsole=midsole_material(), knit=knit_material(), lining=lining_material(),
                tongue=tongue_material(None), tpu=tpu_material(True), tpu_plain=tpu_material(None), hole=hole_material(),
                lace=lace_material(), heeltab=heeltab_material(), insole=insole_material(),
                badge=badge_material(), evasense=evasense_material())
