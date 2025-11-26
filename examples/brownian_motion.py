import numpy as np

from temporalviz import visualize_dynamics

_anim = None  # keep a global ref so it isn't garbage-collected

def main():
    import os

    T, N = 300, 50
    steps = 0.12 * np.random.randn(T, N, 2)
    pos = steps.cumsum(axis=0)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "brownian_motion.mp4")
    global _anim
    _anim = visualize_dynamics(
        {"backend": "matplotlib", "interval": 30, "title": "Brownian Motion"},
        pos,
        save_path=out_path,
    )
    print(f"Saved animation to: {out_path}")


if __name__ == "__main__":
    main()

