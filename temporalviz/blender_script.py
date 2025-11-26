import bpy
import numpy as np
import sys
import json
from pathlib import Path
import math
from mathutils import Vector
from mathutils import Matrix


def _load_inputs():
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 1:
        raise RuntimeError("Expected at least the npz data path after --")
    data_path = Path(args[0])
    opts_path = Path(args[1]) if len(args) >= 2 else None

    with np.load(data_path) as data:
        pos = data["pos"]
        ori = data["ori"]
        node_colors = data["node_colors"] if "node_colors" in data else None
        node_sizes = data["node_sizes"] if "node_sizes" in data else None
        edges = data["edges"] if "edges" in data else None
        edge_colors = data["edge_colors"] if "edge_colors" in data else None

    options = {}
    if opts_path and opts_path.exists():
        options = json.loads(Path(opts_path).read_text())
    return pos, ori, node_colors, node_sizes, edges, edge_colors, options


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


def _create_primitive(shape: str = "sphere", scale: float = 0.1):
    shape = (shape or "sphere").lower()
    if shape == "cone":
        # Allow optional cone parameters via scene custom properties set in options
        r1 = float(bpy.context.scene.get("tz_cone_radius1", 1.0))
        r2 = float(bpy.context.scene.get("tz_cone_radius2", 0.0))
        depth = float(bpy.context.scene.get("tz_cone_depth", 2.0))
        bpy.ops.mesh.primitive_cone_add(radius1=r1, radius2=r2, depth=depth)
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    obj = bpy.context.object
    obj.scale = (scale, scale, scale)
    return obj


def main():
    pos, ori, node_colors, node_sizes, edges, edge_colors, options = _load_inputs()
    T, N = pos.shape[:2]
    D = pos.shape[-1]
    if D != 3:
        # Promote to 3D by padding z=0 for 2D inputs
        pad = np.zeros(pos.shape[:-1] + (3,), dtype=float)
        k = min(3, D)
        pad[..., :k] = pos[..., :k]
        pos = pad
    # Normalize node_sizes to shape (T, N) if provided
    if node_sizes is not None:
        ns = np.array(node_sizes, dtype=float)
        try:
            if ns.ndim == 0:
                ns = np.full((T, N), float(ns))
            elif ns.ndim == 1:
                if ns.shape[0] == N:
                    ns = np.broadcast_to(ns[None, :], (T, N))
                elif ns.shape[0] == T:
                    ns = np.broadcast_to(ns[:, None], (T, N))
                elif ns.shape[0] == 1:
                    ns = np.full((T, N), float(ns[0]))
                else:
                    ns = np.broadcast_to(ns, (T, N))
            elif ns.ndim == 2:
                if ns.shape == (T, N):
                    pass
                elif ns.shape == (T, 1):
                    ns = np.broadcast_to(ns, (T, N))
                elif ns.shape == (1, N):
                    ns = np.broadcast_to(ns, (T, N))
                elif ns.shape == (1, 1):
                    ns = np.full((T, N), float(ns[0, 0]))
                else:
                    ns = np.broadcast_to(ns, (T, N))
            else:
                ns = np.full((T, N), float(ns.flat[0]))
        except Exception:
            ns = np.full((T, N), float(ns.flat[0]))
        node_sizes = ns

    _clear_scene()

    # Scene basics
    scene = bpy.context.scene
    # Robust engine selection across Blender versions
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "CYCLES"
    # Compute bounds and center to frame the scene
    flat = pos.reshape(-1, 3)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    center = 0.5 * (mn + mx)
    extents = mx - mn
    radius = float(max(extents.tolist() + [1.0]))  # Ensure non-zero

    # Target empty at center for camera to track
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(map(float, center)))
    target = bpy.context.object

    # Camera set-up
    if not scene.camera:
        bpy.ops.object.camera_add(location=(float(center[0]), float(center[1] - (3.0 * radius + 2.0)), float(center[2] + (1.5 * radius + 0.5))))
        scene.camera = bpy.context.object
    cam = scene.camera
    cam.data.lens = 35.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000.0
    # Make camera look at the target
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"  # Cameras look along -Z in Blender
    track.up_axis = "UP_Y"

    # Lighting
    bpy.ops.object.light_add(type="SUN", location=(float(center[0] + radius), float(center[1] - radius), float(center[2] + 2.0 * radius)))
    sun = bpy.context.object
    sun.data.energy = 5.0
    # Add a gentle world background so completely black frames are avoided
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    if hasattr(world, "use_nodes") and world.use_nodes:
        # If node-based, set background strength low
        nodes = world.node_tree.nodes
        bg = nodes.get("Background")
        if bg and hasattr(bg.inputs[1], "default_value"):
            bg.inputs[1].default_value = 0.2  # strength
    else:
        # Simple color fallback
        world.color = (0.05, 0.05, 0.06)

    # Optional primitive color (RGB floats 0-1)
    prim_color = options.get("primitive_color")
    if isinstance(prim_color, (list, tuple)) and len(prim_color) >= 3:
        prim_color = tuple(float(c) for c in prim_color[:3])
    else:
        prim_color = None
    primitive_shape = options.get("primitive_shape", "sphere")
    # Stash cone parameters on scene so _create_primitive can read them
    if primitive_shape.lower() == "cone":
        if "cone_radius1" in options:
            scene["tz_cone_radius1"] = float(options["cone_radius1"])
        if "cone_radius2" in options:
            scene["tz_cone_radius2"] = float(options["cone_radius2"])
        if "cone_depth" in options:
            scene["tz_cone_depth"] = float(options["cone_depth"])

    def _ensure_material(name: str, color_rgb):
        # Create or get material
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        # Ensure node tree has Principled -> Material Output link
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links
        # Find nodes by type for robustness across language/localization
        principled = None
        output = None
        for n in nodes:
            if n.type == "BSDF_PRINCIPLED":
                principled = n
            elif n.type == "OUTPUT_MATERIAL":
                output = n
        if principled is None:
            principled = nodes.new("ShaderNodeBsdfPrincipled")
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")
        # Link surface
        # Avoid duplicate links: ensure one link from principled BSDF to output surface
        need_link = True
        for l in links:
            if l.from_node == principled and l.to_node == output and getattr(l.to_socket, "name", "") == "Surface":
                need_link = False
                break
        if need_link:
            # Clear existing inputs to Surface for determinism
            for l in list(links):
                if l.to_node == output and getattr(l.to_socket, "name", "") == "Surface":
                    links.remove(l)
            links.new(principled.outputs.get("BSDF"), output.inputs.get("Surface"))
        # Set color
        if color_rgb is not None and hasattr(principled.inputs[0], "default_value"):
            r, g, b = color_rgb
            principled.inputs[0].default_value = (r, g, b, 1.0)
        mat.blend_method = "OPAQUE"
        return mat

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
            obj = _create_primitive(primitive_shape, scale=scales[i])
            if prim_color is not None:
                mat = _ensure_material("PrimitiveColor", prim_color)
                if obj.data and hasattr(obj.data, "materials"):
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)
        obj.name = f"Agent_{i:03d}"
        objects.append(obj)

    # Create edge curve objects if edges provided
    edge_objs = []
    bevel = float(options.get("edge_bevel", max(0.01, 0.02 * radius)))
    if edges is not None:
        edges = np.asarray(edges, dtype=int)
        for e_idx, (i, j) in enumerate(edges):
            curve = bpy.data.curves.new(name=f"EdgeCurve_{e_idx}", type="CURVE")
            curve.dimensions = "3D"
            curve.fill_mode = "FULL"
            curve.bevel_depth = bevel
            spline = curve.splines.new(type="POLY")
            spline.points.add(1)  # total 2 points
            obj_curve = bpy.data.objects.new(f"Edge_{int(i)}_{int(j)}", curve)
            bpy.context.collection.objects.link(obj_curve)
            # Assign color material if provided static
            if edge_colors is not None and edge_colors.ndim == 2:
                col = edge_colors[e_idx]
                if col.shape[0] >= 3:
                    mat = _ensure_material(f"EdgeColor_{e_idx}", (float(col[0]), float(col[1]), float(col[2])))
                    obj_curve.data.materials.append(mat)
            edge_objs.append((obj_curve, spline))

    # Animate
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(0, T - 1)
    scene.render.fps = int(options.get("fps", 24))

    # If using 3D direction vectors, map local +Z (cone axis) to target direction
    use_dir_vectors = ori is not None and ori.shape[-1] == 3

    for t in range(T):
        scene.frame_set(t)
        for i, obj in enumerate(objects):
            obj.location = tuple(float(x) for x in pos[t, i])
            obj.keyframe_insert(data_path="location")
            # Orientation handling
            if ori.shape[-1] == 4:
                q = ori[t, i]
                obj.rotation_mode = "QUATERNION"
                # Assume quaternion as (w,x,y,z)
                obj.rotation_quaternion = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                obj.keyframe_insert("rotation_quaternion")
            elif ori.shape[-1] == 3:
                # Map local +Z (cone axis) to desired direction using rotation_difference
                d = Vector((float(ori[t, i, 0]), float(ori[t, i, 1]), float(ori[t, i, 2])))
                if d.length > 1e-12:
                    v_from = Vector((0.0, 0.0, 1.0))
                    v_to = d.normalized()
                    q = v_from.rotation_difference(v_to)
                    obj.rotation_mode = "QUATERNION"
                    obj.rotation_quaternion = (q.w, q.x, q.y, q.z)
                    obj.keyframe_insert("rotation_quaternion")
            # Size animation
            if node_sizes is not None:
                if node_sizes.ndim == 1:
                    s = float(node_sizes[i])
                else:
                    s = float(node_sizes[t, i])
                obj.scale = (s * scales[i], s * scales[i], s * scales[i])
                obj.keyframe_insert(data_path="scale")
            # Color animation
            if node_colors is not None:
                if node_colors.ndim >= 3:
                    col = node_colors[t, i]
                else:
                    col = node_colors[i]
                if col.shape[0] >= 3 and obj.data and obj.data.materials:
                    mat = obj.data.materials[0]
                    if getattr(mat, "use_nodes", False):
                        nodes = mat.node_tree.nodes
                        principled = None
                        for n in nodes:
                            if n.type == "BSDF_PRINCIPLED":
                                principled = n
                                break
                        if principled:
                            r, g, b = float(col[0]), float(col[1]), float(col[2])
                            principled.inputs[0].default_value = (r, g, b, 1.0)
                            principled.inputs[0].keyframe_insert("default_value")
        # Update edges per frame
        if edge_objs:
            for e_idx, ((obj_curve, spline)) in enumerate(edge_objs):
                (i, j) = edges[e_idx]
                p_i = pos[t, int(i)]
                p_j = pos[t, int(j)]
                spline.points[0].co = (float(p_i[0]), float(p_i[1]), float(p_i[2]), 1.0)
                spline.points[1].co = (float(p_j[0]), float(p_j[1]), float(p_j[2]), 1.0)
                obj_curve.data.keyframe_insert(data_path="splines[0].points[0].co")
                obj_curve.data.keyframe_insert(data_path="splines[0].points[1].co")
                # Edge color per-frame
                if edge_colors is not None and edge_colors.ndim == 3:
                    col = edge_colors[t, e_idx]
                    if col.shape[0] >= 3:
                        mat = None
                        if obj_curve.data.materials:
                            mat = obj_curve.data.materials[0]
                        else:
                            mat = _ensure_material(f"EdgeColor_{e_idx}", (float(col[0]), float(col[1]), float(col[2])))
                            obj_curve.data.materials.append(mat)
                        if getattr(mat, "use_nodes", False):
                            nodes = mat.node_tree.nodes
                            principled = None
                            for n in nodes:
                                if n.type == "BSDF_PRINCIPLED":
                                    principled = n
                                    break
                            if principled:
                                r, g, b = float(col[0]), float(col[1]), float(col[2])
                                principled.inputs[0].default_value = (r, g, b, 1.0)
                                principled.inputs[0].keyframe_insert("default_value")

    save_path = options.get("save_path")
    if save_path:
        # Configure output
        scene.render.filepath = save_path
        scene.render.image_settings.file_format = "FFMPEG"
        # Container/codec
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        # Quality/preset
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        # Render the whole timeline to video
        bpy.ops.render.render(animation=True)
        print(f"Rendered animation to: {save_path}")

    # Optionally save a .blend file for further editing in Blender
    save_blend = bool(options.get("save_blend", False))
    blend_path = options.get("blend_path")
    if save_blend or blend_path:
        # Derive a sensible default path if none specified
        if not blend_path or not isinstance(blend_path, str) or not blend_path.strip():
            # If we rendered a video, put the .blend next to it; otherwise use cwd
            if save_path:
                base = Path(save_path)
                stem = base.stem or "scene"
                blend_path = str(base.with_name(stem + ".blend"))
            else:
                blend_path = "temporalviz_output.blend"
        # Ensure .blend extension
        if not str(blend_path).lower().endswith(".blend"):
            blend_path = str(Path(blend_path).with_suffix(".blend"))
        # Make sure directory exists
        try:
            Path(blend_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"Saved Blender file to: {blend_path}")


if __name__ == "__main__":
    main()