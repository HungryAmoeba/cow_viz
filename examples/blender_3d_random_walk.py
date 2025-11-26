import numpy as np
import os

from temporalviz import visualize_dynamics


def main():
    T, N = 180, 30
    steps = 0.05 * np.random.randn(T, N, 3)
    pos = steps.cumsum(axis=0)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "blender_3d_random_walk.mp4")

    cfg = {
        "backend": "blender",
        "fps": 24,
        # If Blender isn't discovered automatically on macOS, uncomment and set your path:
        # "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender",
        "save_path": out_path,
    }
    visualize_dynamics(cfg, pos)
    print(f"Rendered Blender animation to: {out_path}")


if __name__ == "__main__":
    main()

