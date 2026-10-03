"""Bake the Cycles materials to PBR textures and export a web-ready glTF binary (.glb).

usage: python export_glb.py -- <in.blend> <out.glb> [tex_scale]
Every visible mesh is converted (modifiers applied), decimated where dense, UV-unwrapped,
baked (base colour + tangent-space normal) and re-assigned a plain glTF PBR material.
"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy, bmesh

args = sys.argv[sys.argv.index('--') + 1:]
SRC = os.path.join(HERE, args[0])
DST = os.path.join(HERE, args[1])
TSCALE = float(args[2]) if len(args) > 2 else 1.0
BAKEDIR = os.path.join(os.path.dirname(DST), 'glb_bake')
os.makedirs(BAKEDIR, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=SRC)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 8
sc.cycles.use_denoising = False
sc.render.bake.margin = 6

# name: (texture size, decimate ratio, roughness, sheen)
PLAN = {
    'Upper_Knit': ((4096, 1024), 0.45, 0.78, 0.5),     # perimeter x wall-height UV: keep texels ~isotropic
    'Midsole': (2048, 0.30, 0.58, 0.0),
    'Collar_Padding': (1024, 1.0, 0.72, 0.6),
    'Tongue': (1024, 0.7, 0.70, 0.5),
    'Eyestay_TPU': (1024, 0.25, 0.42, 0.0),
    'Laces': (1024, 0.6, 0.74, 0.6),
    'Logo_Badge': (1024, 0.06, 0.33, 0.0),
    'Insole': (1024, 0.5, 0.85, 0.3),
    'Heel_Pull_Tab': (512, 1.0, 0.60, 0.5),
    'Throat_Binding': (256, 1.0, 0.72, 0.6),
    'Eyelet_Rims': (256, 1.0, 0.42, 0.0),
    'Eyelet_Holes': (128, 1.0, 0.90, 0.0),
    'EVASENSE_Print': (512, 1.0, 0.50, 0.0),
}

deps = bpy.context.evaluated_depsgraph_get()
root = bpy.data.objects['Olympikus_Lumi_Root']
out_objs = []
for name, (tex, ratio, rough, sheen) in PLAN.items():
    src = bpy.data.objects.get(name)
    if src is None:
        continue
    tw, th = (tex if isinstance(tex, tuple) else (tex, tex))
    tw, th = max(128, int(tw * TSCALE)), max(128, int(th * TSCALE))
    # evaluated copy with modifiers applied
    # inner shell / rim of solidified parts -> their own slot so they never overwrite the outer bake
    n_slots = len(src.data.materials)
    sol_mods = [m for m in src.modifiers if m.type == 'SOLIDIFY']
    saved_off = [(m, m.material_offset, m.material_offset_rim) for m in sol_mods]
    if sol_mods and n_slots == 1:
        src.data.materials.append(src.data.materials[0])
        for m in sol_mods:
            m.material_offset = 1
            m.material_offset_rim = 1
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    ev = src.evaluated_get(deps)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=deps)
    for m, o1, o2 in saved_off:
        m.material_offset, m.material_offset_rim = o1, o2
    if sol_mods and n_slots == 1:
        src.data.materials.pop(index=1)
    ob = bpy.data.objects.new('GLB_' + name, me)
    sc.collection.objects.link(ob)
    ob.matrix_world = src.matrix_world.copy()
    # materials: single-user copies
    me.materials.clear()
    for m in src.data.materials:
        me.materials.append(m.copy() if m else None)
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    if ratio < 1.0:
        dm = ob.modifiers.new('dec', 'DECIMATE')
        dm.ratio = ratio
        dm.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier='dec')
    # keep the build UVs when present (tongue/eyestay/laces/badge/decal); else unwrap
    need_unwrap = name in ('Midsole', 'Collar_Padding', 'Insole', 'Heel_Pull_Tab',
                           'Throat_Binding', 'Eyelet_Rims', 'Eyelet_Holes')
    if need_unwrap or len(me.uv_layers) == 0:
        keep = [uv.name for uv in me.uv_layers]
        uvb = me.uv_layers.new(name='BakeUV')
        me.uv_layers.active = uvb
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.006, area_weight=0.0)
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        uvb = me.uv_layers.active
        uvb.name = 'BakeUV'
    # images + active texture node in every material slot
    img_c = bpy.data.images.new(f'{name}_basecolor', tw, th, alpha=False)
    img_n = bpy.data.images.new(f'{name}_normal', tw, th, alpha=False)
    img_n.colorspace_settings.name = 'Non-Color'
    nodes_c = []
    img_dummy = bpy.data.images.new(f'{name}_dummy', 64, 64, alpha=False)
    for si, m in enumerate(me.materials):
        if m is None:
            continue
        nt = m.node_tree
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'BakeUV'
        tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img_c if si == 0 else img_dummy
        nt.links.new(uvn.outputs[0], tn.inputs['Vector'])
        nt.nodes.active = tn
        if si == 0:
            nodes_c.append(tn)
    # make the bake UV the render-active one so the bake writes into it
    for uv in me.uv_layers:
        uv.active_render = (uv.name == 'BakeUV')
    me.uv_layers.active = me.uv_layers['BakeUV']
    sc.cycles.bake_type = 'DIFFUSE'
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.use_pass_color = True
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, use_clear=True, margin=6)
    img_c.filepath_raw = os.path.join(BAKEDIR, f'{name}_basecolor.png')
    img_c.file_format = 'PNG'
    img_c.save()
    for tn in nodes_c:
        tn.image = img_n
    bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', use_clear=True, margin=6)
    img_n.filepath_raw = os.path.join(BAKEDIR, f'{name}_normal.png')
    img_n.file_format = 'PNG'
    img_n.save()
    # plain glTF material
    alpha_mat = (name == 'EVASENSE_Print')
    gm = bpy.data.materials.new('GLB_' + name)
    gm.use_nodes = True
    b = gm.node_tree.nodes['Principled BSDF']
    nt = gm.node_tree
    uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'BakeUV'
    tc = nt.nodes.new('ShaderNodeTexImage'); tc.image = img_c
    nt.links.new(uvn.outputs[0], tc.inputs['Vector'])
    nt.links.new(tc.outputs['Color'], b.inputs['Base Color'])
    tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img_n
    nt.links.new(uvn.outputs[0], tn.inputs['Vector'])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nm.uv_map = 'BakeUV'
    nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    b.inputs['Roughness'].default_value = rough
    b.inputs['Sheen Weight'].default_value = sheen
    if alpha_mat:
        src_m = src.data.materials[0]
        # reuse the decal alpha
        for n_ in src_m.node_tree.nodes:
            if n_.type == 'TEX_IMAGE':
                ta = nt.nodes.new('ShaderNodeTexImage'); ta.image = n_.image
                nt.links.new(uvn.outputs[0], ta.inputs['Vector'])
                nt.links.new(ta.outputs['Color'], b.inputs['Alpha'])
                b.inputs['Base Color'].default_value = (0.012, 0.012, 0.014, 1)
                nt.links.remove(nt.links[[l.to_socket for l in nt.links].index(b.inputs['Base Color'])]) if b.inputs['Base Color'].is_linked else None
                break
    # secondary slots (inner shell, rim, lining) -> plain colour taken from the source material
    extra = []
    for si, m in enumerate(list(me.materials)[1:], start=1):
        pm = bpy.data.materials.new(f'GLB_{name}_inner{si}')
        pm.use_nodes = True
        pb = pm.node_tree.nodes['Principled BSDF']
        col = (0.8, 0.8, 0.8, 1)
        for n_ in (m.node_tree.nodes if m else []):
            if n_.type == 'BSDF_PRINCIPLED' and not n_.inputs['Base Color'].is_linked:
                col = tuple(n_.inputs['Base Color'].default_value)
                break
        if name == 'Upper_Knit':
            col = (*__import__('bl_util').hex_lin('#A2A9C8'), 1)
        pb.inputs['Base Color'].default_value = col
        pb.inputs['Roughness'].default_value = rough
        extra.append(pm)
    me.materials.clear()
    me.materials.append(gm)
    for pm in extra:
        me.materials.append(pm)
    # drop the other UV maps (keep only BakeUV)
    for nm_ in [uv.name for uv in me.uv_layers if uv.name != 'BakeUV' and not uv.name.startswith('.')]:
        if nm_ in me.uv_layers:
            me.uv_layers.remove(me.uv_layers[nm_])
    for a in list(me.attributes):
        if a.name in ('u_len', 'tau', 'ts', 'tt', 'tfront', 'tarc', 'ty', 'es_s', 'es_t', 'tab_side'):
            try:
                me.attributes.remove(a)
            except Exception:
                pass
    out_objs.append(ob)
    print('baked', name, len(me.vertices), 'verts', tw, 'x', th, 'px')

# export only the baked objects
bpy.ops.object.select_all(action='DESELECT')
for o in out_objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = out_objs[0]
kw = dict(filepath=DST, export_format='GLB', use_selection=True, export_apply=True,
          export_texcoords=True, export_normals=True, export_tangents=False,
          export_materials='EXPORT', export_image_format='JPEG', export_jpeg_quality=88,
          export_yup=True, export_cameras=False, export_lights=False)
try:
    bpy.ops.export_scene.gltf(**kw, export_draco_mesh_compression_enable=True,
                              export_draco_mesh_compression_level=7, export_draco_position_quantization=14,
                              export_draco_normal_quantization=10, export_draco_texcoord_quantization=12)
    print('exported with draco')
except Exception as e:
    print('draco export failed:', e)
    bpy.ops.export_scene.gltf(**kw)
print('GLB', DST, os.path.getsize(DST) / 1e6, 'MB')
