import os
import numpy as np

from temporalviz import visualize_dynamics


def helix(T, radius=1.0, pitch=0.2, chirality=1.0):
    t = np.linspace(0, 4 * np.pi, T)

    # Position on the helix
    x = radius * np.cos(chirality * t)
    y = radius * np.sin(chirality * t)
    z = pitch * t
    pos = np.stack([x, y, z], axis=-1)  # (T, 3)

    # Analytic tangent (derivative wrt t)
    dx = -radius * chirality * np.sin(chirality * t)
    dy =  radius * chirality * np.cos(chirality * t)
    dz =  pitch * np.ones_like(t)

    dir_vec = np.stack([dx, dy, dz], axis=-1)
    dir_vec /= (np.linalg.norm(dir_vec, axis=-1, keepdims=True) + 1e-12)

    return pos, dir_vec
    
def main():
    T = 200
    # Two helices with opposite chirality
    pos0, dir0 = helix(T, radius=1.0, pitch=0.15, chirality=+1.0)
    pos1, dir1 = helix(T, radius=1.0, pitch=0.15, chirality=-1.0)

    # Stack into (T, N, 3)
    pos = np.stack([pos0, pos1], axis=1)
    # Orientations:
    # - Cone 0: tangent (already computed as dir0)
    # - Cone 2: radial (points outward in XY plane)
    radial1 = pos1.copy()
    radial1[..., 2] = 0.0
    radial1 /= (np.linalg.norm(radial1, axis=-1, keepdims=True) + 1e-12)
    ori = np.stack([dir0, radial1], axis=1)  # (T,2,3) direction vectors

    # Colors per node (constant over time)
    col0 = np.array([0.2, 0.8, 1.0])  # cyan
    col1 = np.array([1.0, 0.4, 0.2])  # orange
    node_colors = np.stack([np.tile(col0, (T, 1)), np.tile(col1, (T, 1))], axis=1)

    # Sizes: gentle pulsation around 1.0
    t = np.linspace(0, 2 * np.pi, T)
    node_sizes = 1.0 + 0.2 * np.sin(t)[:, None]  # (T,1) broadcast to (T,2)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "blender_3d_cones_helix.mp4")

    cfg = {
        "backend": "blender",
        "fps": 24,
        "save_path": out_path,
        "primitive_shape": "cone",
        "obj_scales": 0.25,
        # Make cones pointier by shrinking base radius and increasing depth
        "cone_radius1": 0.3,
        "cone_radius2": 0.0,
        "cone_depth": 3.0,
        # "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender",
    }

    visualize_dynamics(
        cfg,
        pos,
        ori=ori,
        node_colors=node_colors,
        node_sizes=node_sizes,
    )
    print(f"Rendered Blender cones helix demo to: {out_path}")


if __name__ == "__main__":
    main()

