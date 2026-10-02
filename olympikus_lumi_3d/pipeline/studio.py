"""Product studio for the Lumi digital twin: seamless white set, softbox lighting, hero cameras.

usage: python studio.py -- <blend> <views comma> <res_x> <samples> [outdir] [exposure]
"""
import sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import bpy
from mathutils import Vector, Euler, Matrix
import bl_util as bu
import shoe_geom as g

args = sys.argv[sys.argv.index('--') + 1:]
BLEND = os.path.join(HERE, args[0])
VIEWS = args[1].split(',')
RESX = int(args[2]) if len(args) > 2 else 1600
SAMPLES = int(args[3]) if len(args) > 3 else 128
OUTDIR = os.path.join(HERE, args[4]) if len(args) > 4 else os.path.join(HERE, 'renders')
EXPOSURE = float(args[5]) if len(args) > 5 else 0.0
os.makedirs(OUTDIR, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = SAMPLES
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.008
sc.cycles.use_denoising = True
sc.cycles.denoiser = 'OPENIMAGEDENOISE'
sc.cycles.max_bounces = 10
sc.cycles.diffuse_bounces = 4
sc.cycles.glossy_bounces = 4
sc.cycles.transmission_bounces = 4
sc.cycles.caustics_reflective = False
sc.cycles.caustics_refractive = False
sc.cycles.blur_glossy = 1.0
sc.render.film_transparent = True
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.view_settings.exposure = EXPOSURE
sc.view_settings.gamma = 1.0
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'

root = bpy.data.objects['Olympikus_Lumi_Root']
for o in bpy.data.objects:
    if o.name.startswith('Eyelet_Cutters'):
        o.hide_render = True

# ---------------------------------------------------------------- world: white, camera rays clip to pure white
world = bpy.data.worlds.new('Studio_World')
sc.world = world
world.use_nodes = True
nt = world.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new('ShaderNodeOutputWorld')
bg_light = nt.nodes.new('ShaderNodeBackground')
bg_light.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1)
bg_light.inputs['Strength'].default_value = 0.40
bg_cam = nt.nodes.new('ShaderNodeBackground')
bg_cam.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1)
bg_cam.inputs['Strength'].default_value = 4.0
lp = nt.nodes.new('ShaderNodeLightPath')
mix = nt.nodes.new('ShaderNodeMixShader')
nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Fac'])
nt.links.new(bg_light.outputs[0], mix.inputs[1])
nt.links.new(bg_cam.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], out.inputs['Surface'])

# ---------------------------------------------------------------- seamless floor (large disc, white)
bpy.ops.mesh.primitive_circle_add(vertices=128, radius=6.0, fill_type='NGON', location=(0, 0, 0))
floor = bpy.context.object
floor.name = 'Studio_Floor'
fm = bpy.data.materials.new('Studio_Floor_White')
fm.use_nodes = True
fb = fm.node_tree.nodes['Principled BSDF']
fb.inputs['Base Color'].default_value = (0.92, 0.92, 0.92, 1)
fb.inputs['Roughness'].default_value = 0.85
fb.inputs['Specular IOR Level'].default_value = 0.2
floor.data.materials.append(fm)
floor.is_shadow_catcher = True


def area(name, loc, target, size, energy, shape='RECTANGLE', size_y=None, spread=180):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = energy
    ld.spread = math.radians(spread)
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    ob.location = loc
    bu.look_at(ob, target)
    return ob


T0 = Vector((0.0, 0.0, 0.055))
# soft light tent + key
area('Softbox_Top', (0.0, 0.0, 1.05), (0, 0, 0), 1.3, 5.6, size_y=0.9)
area('Softbox_Key', (0.55, -0.85, 0.75), T0, 0.9, 9.0, size_y=0.9)
area('Softbox_Fill', (-0.75, -0.55, 0.45), T0, 1.0, 2.2, size_y=0.8)
area('Softbox_Medial', (0.1, 0.95, 0.55), T0, 1.0, 3.4, size_y=0.8)
area('Strip_Rim', (-0.85, 0.35, 0.65), T0, 0.25, 3.8, size_y=1.2)

# ---------------------------------------------------------------- cameras
CAMS = {
    # name: (location, target, lens mm, aspect w/h)
    'side_lateral': ((0.0, -1.55, 0.135), (0.0, 0.0, 0.058), 125, 1.6),
    'side_medial': ((0.0, 1.55, 0.135), (0.0, 0.0, 0.058), 125, 1.6),
    'q34_front': ((0.66, -0.80, 0.42), (0.008, 0.0, 0.050), 120, 1.5),
    'q34_rear': ((-0.74, -0.62, 0.40), (-0.010, 0.0, 0.055), 120, 1.5),
    'top_down': ((0.0, 0.0, 1.30), (0.0, 0.0, 0.0), 120, 0.68),
    'macro_logo': ((-0.020, -0.40, 0.095), (-0.021, -0.040, 0.068), 135, 1.5),
    'macro_tongue': ((0.10, -0.17, 0.27), (-0.004, -0.004, 0.105), 135, 1.5),
    'macro_heel': ((-0.32, -0.24, 0.20), (-0.112, -0.004, 0.075), 135, 1.5),
    'macro_knit': ((0.090, -0.21, 0.080), (0.065, -0.043, 0.050), 135, 1.5),
    'sole': ((0.0, 0.0, 1.30), (0.0, 0.0, 0.0), 120, 0.68),
}


def setup_cam(name):
    loc, tgt, lens, aspect = CAMS[name]
    cd = bpy.data.cameras.new('CAM_' + name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.sensor_fit = 'HORIZONTAL' if aspect >= 1 else 'VERTICAL'
    if aspect < 1:
        cd.sensor_height = 36
    cd.clip_start = 0.01
    cd.clip_end = 50
    cam = bpy.data.objects.new('CAM_' + name, cd)
    sc.collection.objects.link(cam)
    cam.location = loc
    bu.look_at(cam, tgt)
    if name in ('top_down', 'sole'):
        cam.rotation_euler = Euler((0, 0, math.pi / 2), 'XYZ')   # toe to the right... rotate below
        cam.rotation_euler = Euler((0, 0, 0), 'XYZ')
    if name.startswith('macro'):
        cd.dof.use_dof = True
        cd.dof.focus_distance = (Vector(loc) - Vector(tgt)).length
        cd.dof.aperture_fstop = 5.6
    else:
        cd.dof.use_dof = False
    return cam, aspect


def frame_ortho_fit(cam, aspect, margin=1.12):
    """adjust focal length so the shoe fills the frame (perspective cams)."""
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.parent == root and o.visible_get() and not o.hide_render:
            for c in o.bound_box:
                pts.append(o.matrix_world @ Vector(c))
    M = cam.matrix_world.inverted()
    pc = [M @ p for p in pts]
    xs = [p.x / -p.z for p in pc]
    ys = [p.y / -p.z for p in pc]
    half_w = max(abs(min(xs)), abs(max(xs)))
    half_h = max(abs(min(ys)), abs(max(ys)))
    cd = cam.data
    if aspect >= 1:
        # horizontal sensor 36 mm : tan(half fov) = 18/lens
        need = max(half_w, half_h * aspect) * margin
        cd.lens = 18.0 / need
    else:
        need = max(half_h, half_w / aspect) * margin
        cd.lens = 18.0 / need


def match_cam(v):
    """cameras matched to the reference photos (ortho sides, perspective top) - see compare.py"""
    S = g.S
    if v in ('match_lat', 'match_med'):
        side = S['side1'] if v == 'match_lat' else S['side2']
        W, H = 4877, 3024
        Lp = side['x1'] - side['x0']
        cd = bpy.data.cameras.new('CAM_' + v); cd.type = 'ORTHO'; cd.ortho_scale = W / Lp * g.L
        cd.clip_start = 0.01; cd.clip_end = 50
        cam = bpy.data.objects.new('CAM_' + v, cd); sc.collection.objects.link(cam)
        zc = (side['ground'] - H / 2) / Lp * g.L
        if v == 'match_lat':
            cam.location = (g.X0 + (W / 2 - side['x0']) / Lp * g.L, -1.0, zc); cam.rotation_euler = (math.pi / 2, 0, 0)
        else:
            cam.location = (g.X0 + (1 - (W / 2 - side['x0']) / Lp) * g.L, 1.0, zc); cam.rotation_euler = (math.pi / 2, 0, math.pi)
        return cam, W / H
    tv = S['topview']; W, H = 2949, 4258
    Lt = tv['row_toe'] - tv['row_heel']
    D = 1.94 * g.L
    u_c = (H / 2 - tv['row_heel']) / Lt
    slope = (tv['axis_c_toe'] - tv['axis_c_heel']) / (0.985 - 0.08)
    axis_c = tv['axis_c_heel'] + slope * (u_c - 0.08)
    cd = bpy.data.cameras.new('CAM_' + v); cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 24.0
    cd.lens = cd.sensor_height * (D - 0.10 * g.L) / (H * g.L / Lt)
    cd.clip_start = 0.01; cd.clip_end = 50
    cam = bpy.data.objects.new('CAM_' + v, cd); sc.collection.objects.link(cam)
    cam.location = (g.X0 + u_c * g.L, (W / 2 - axis_c) / Lt * g.L / g.WIDTH_SCALE, D)
    cam.rotation_euler = (0, 0, math.pi / 2)
    return cam, W / H


if VIEWS == ['save']:
    # package mode: keep every hero camera in the scene and save a self-contained studio file
    import shutil
    dst_blend = os.path.join(OUTDIR, 'olympikus_lumi.blend')
    texdir = os.path.join(OUTDIR, 'texturas')
    os.makedirs(texdir, exist_ok=True)
    for o in list(bpy.data.objects):
        if o.name.startswith('Eyelet_Cutters'):
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        if c.name == 'Cutters':
            bpy.data.collections.remove(c)
    first = None
    for v in ('q34_front', 'q34_rear', 'side_lateral', 'side_medial', 'top_down', 'macro_logo', 'macro_tongue', 'macro_knit', 'macro_heel'):
        cam, aspect = setup_cam(v)
        if v == 'top_down':
            cam.rotation_euler = Euler((0, 0, math.pi / 2), 'XYZ')
        if not v.startswith('macro'):
            frame_ortho_fit(cam, aspect, margin=1.10 if v != 'top_down' else 1.06)
        first = first or cam
    sc.camera = first
    sc.render.resolution_x = 2400
    sc.render.resolution_y = 1600
    sc.render.image_settings.color_mode = 'RGBA'
    for im in bpy.data.images:
        if im.source == 'FILE' and im.filepath:
            src = bpy.path.abspath(im.filepath)
            if os.path.exists(src):
                dst = os.path.join(texdir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                im.filepath = dst
    bpy.ops.wm.save_as_mainfile(filepath=dst_blend, compress=True, relative_remap=True)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=dst_blend, compress=True)
    print('PACKAGED', dst_blend)
    sys.exit(0)

saved = {}
for v in VIEWS:
    # pose the shoe
    root.rotation_euler = (0, 0, 0)
    root.location = (0, 0, 0)
    floor.hide_render = False
    if v in ('top_down',):
        root.rotation_euler = (0, 0, math.radians(-90))     # heel at the top, toe at the bottom (as the reference)
    if v == 'sole':
        root.rotation_euler = (math.pi, 0, math.radians(90))
        root.location = (0, 0, 0.140)
    bpy.context.view_layer.update()
    if v.startswith('match_'):
        cam, aspect = match_cam(v)
    else:
        cam, aspect = setup_cam(v)
    sc.camera = cam
    sc.render.resolution_x = RESX
    sc.render.resolution_y = int(round(RESX / aspect))
    sc.render.resolution_percentage = 100
    if not v.startswith('macro') and not v.startswith('match_'):
        frame_ortho_fit(cam, aspect, margin=1.10 if v not in ('top_down', 'sole') else 1.06)
    f = os.path.join(OUTDIR, f'{v}_rgba.png')
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.filepath = f
    bpy.ops.render.render(write_still=True)
    # composite over seamless white (shadow catcher alpha -> soft contact shadow)
    from PIL import Image
    im = Image.open(f).convert('RGBA')
    a = np.asarray(im).astype(np.float32) / 255.0
    rgb, al = a[..., :3], a[..., 3:4]
    outp = rgb + (1 - al) * 1.0          # premultiplied-style over white (Blender saves straight alpha)
    outp = rgb * al + (1 - al) * 1.0
    fo = os.path.join(OUTDIR, f'{v}.png')
    Image.fromarray((np.clip(outp, 0, 1) * 255 + 0.5).astype(np.uint8)).save(fo)
    saved[v] = fo
    print('RENDERED', v, f)
json.dump(saved, open(os.path.join(OUTDIR, '_last.json'), 'w'))
