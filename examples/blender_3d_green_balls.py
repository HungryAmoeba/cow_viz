import numpy as np
import os

from temporalviz import visualize_dynamics


def main():
    T, N = 180, 30
    steps = 0.05 * np.random.randn(T, N, 3)
    pos = steps.cumsum(axis=0)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "blender_3d_green_balls.mp4")

    cfg = {
        "backend": "blender",
        "fps": 24,
        "save_path": out_path,
        # Make spheres green; values in [0,1]
        "primitive_color": [0.15, 0.8, 0.2],
        # Optionally scale the spheres up/down
        #"obj_scales": 0.15,
        # If Blender isn't discovered automatically on macOS, set blender_exec:
        # "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender",
    }
    visualize_dynamics(cfg, pos)
    print(f"Rendered Blender animation to: {out_path}")


if __name__ == "__main__":
    main()

