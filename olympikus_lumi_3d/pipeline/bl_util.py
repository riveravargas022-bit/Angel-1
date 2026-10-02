"""Blender helpers (bpy as a module)."""
import bpy, bmesh
import numpy as np
from mathutils import Vector


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_from(name, V, F, smooth=True, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in np.asarray(V, float)], [], [tuple(int(i) for i in f) for f in F])
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    if smooth:
        me.shade_smooth()
    return ob


def add_float_attr(ob, name, values, domain='POINT'):
    me = ob.data
    at = me.attributes.new(name=name, type='FLOAT', domain=domain)
    at.data.foreach_set('value', np.asarray(values, np.float32).ravel())
    return at


def merge_by_distance(ob, dist=1e-6):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def ortho_cam(name, loc, rot_euler, scale, sensor_fit='HORIZONTAL'):
    cd = bpy.data.cameras.new(name)
    cd.type = 'ORTHO'
    cd.ortho_scale = scale
    cd.sensor_fit = sensor_fit
    cd.clip_start = 0.001
    cd.clip_end = 100
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = rot_euler
    return ob


def persp_cam(name, loc, target, lens=85, sensor=36):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.005
    cd.clip_end = 100
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def look_at(ob, target, roll=0.0):
    d = Vector(target) - ob.location
    q = d.to_track_quat('-Z', 'Y')
    ob.rotation_euler = q.to_euler()
    if roll:
        ob.rotation_euler.rotate_axis('Z', roll)


def principled(name, color=(0.8, 0.8, 0.8), rough=0.5, **kw):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        b.inputs[k].default_value = v
    return m


def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_lin(hx):
    hx = hx.lstrip('#')
    c = np.array([int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return tuple(srgb_to_lin(c))
