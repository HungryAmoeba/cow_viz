import os
import numpy as np

from temporalviz import visualize_dynamics


def main():
    T, N = 180, 20
    # Arrange objects far apart on a large circle, then spin each independently
    radius = 6.0
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_pos = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=-1)  # (N,3)
    pos = np.broadcast_to(base_pos[None, :, :], (T, N, 3)).copy()
    # Optional gentle vertical bob for visual interest
    t = np.linspace(0, 2 * np.pi, T, endpoint=False, dtype=float)
    pos[:, :, 2] = 0.3 * np.sin(t)[:, None]

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "blender_3d_obj_model.mp4")

    # Use the bundled cow model
    # Layout:
    #   examples/data/cow/source/cow/cow.obj
    #   examples/data/cow/source/cow/cow.mtl
    #   examples/data/cow/source/cow/textures/tornado_cow_UV_albedo.png
    data_dir = os.path.join(os.path.dirname(__file__), "data", "cow")
    model_path = os.path.join(data_dir, "source", "cow", "cow.obj")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Expected cow model at '{model_path}'. "
            "Please ensure examples/data/cow/source/cow/cow.obj exists."
        )
    obj_paths = [model_path] * N  # One entry per agent (reuse the same model)

    # Create smooth, independent spins via per-object quaternions (w, x, y, z)
    # Random axis per object, random speed in [0.5, 1.5] rotations over the animation
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(N, 3))
    axes /= (np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12)  # normalize
    rotations = 0.5 + rng.random(N)  # rotations over full sequence
    angles = (2.0 * np.pi) * rotations[None, :] * (np.arange(T, dtype=float)[:, None] / float(max(T - 1, 1)))  # (T,N)
    half = 0.5 * angles
    cos_half = np.cos(half)
    sin_half = np.sin(half)
    ori = np.empty((T, N, 4), dtype=float)
    ori[:, :, 0] = cos_half  # w
    ori[:, :, 1:] = axes[None, :, :] * sin_half[:, :, None]  # x,y,z

    cfg = {
        "backend": "blender",
        "fps": 24,
        "save_path": out_path,
        "obj_paths": obj_paths,
        # Uniform scale for the model(s)
        "obj_scales": 0.1,
        # If Blender isn't discovered automatically on macOS, set blender_exec:
        # "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender",
    }
    visualize_dynamics(cfg, pos, ori=ori)
    print(f"Rendered Blender animation with OBJ model to: {out_path}")


if __name__ == "__main__":
    main()


