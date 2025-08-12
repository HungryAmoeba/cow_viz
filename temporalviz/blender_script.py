import bpy
import numpy as np
import sys
import json
from pathlib import Path


def _load_inputs():
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 1:
        raise RuntimeError("Expected at least the npz data path after --")
    data_path = Path(args[0])
    opts_path = Path(args[1]) if len(args) >= 2 else None

    with np.load(data_path) as data:
        pos = data["pos"]
        ori = data["ori"]

    options = {}
    if opts_path and opts_path.exists():
        options = json.loads(Path(opts_path).read_text())
    return pos, ori, options


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_obj(path: str, scale=1.0):
    ext = Path(path).suffix.lower()
    obj = None
    try:
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)
            obj = bpy.context.view_layer.objects.active
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
            obj = bpy.context.view_layer.objects.active
        else:
            print(f"Unsupported object format: {ext}")
            return None
        if obj:
            obj.scale = (scale, scale, scale)
        return obj
    except Exception as e:
        print(f"Failed to import {path}: {e}")
        return None


def _create_primitive(scale=0.1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=scale)
    return bpy.context.object


def main():
    pos, ori, options = _load_inputs()
    T, N = pos.shape[:2]

    _clear_scene()

    # Scene basics
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add(location=(0, -5, 2))
        bpy.context.scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN", location=(5, 5, 5))

    # Build per-agent objects
    obj_paths = options.get("obj_paths")
    obj_scales = options.get("obj_scales", 0.1)
    if isinstance(obj_scales, (int, float)):
        scales = [float(obj_scales)] * N
    else:
        scales = list(obj_scales) + [0.1] * max(0, N - len(obj_scales))
        scales = scales[:N]

    objects = []
    for i in range(N):
        obj = None
        if obj_paths and i < len(obj_paths) and obj_paths[i]:
            obj = _import_obj(obj_paths[i], scale=scales[i])
        if obj is None:
            obj = _create_primitive(scale=scales[i])
        obj.name = f"Agent_{i:03d}"
        objects.append(obj)

    # Animate
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(0, T - 1)
    scene.render.fps = int(options.get("fps", 24))

    for t in range(T):
        scene.frame_set(t)
        for i, obj in enumerate(objects):
            obj.location = tuple(float(x) for x in pos[t, i])
            obj.keyframe_insert(data_path="location")
            # Orientations are assumed quaternion-like (w,x,y,z)
            if ori.shape[-1] == 4:
                q = ori[t, i]
                obj.rotation_mode = "QUATERNION"
                obj.rotation_quaternion = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
                obj.keyframe_insert("rotation_quaternion")

    save_path = options.get("save_path")
    if save_path:
        # Configure for viewport render; users can adjust to full render if needed
        bpy.context.scene.render.filepath = save_path
        print(f"Animation ready at: {save_path}")


if __name__ == "__main__":
    main()