import os
import numpy as np
import networkx as nx

from temporalviz import visualize_dynamics


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


def main():
    def smooth_vectors_time(vecs: np.ndarray, win: int = 5) -> np.ndarray:
        if win <= 1:
            v = vecs.copy()
        else:
            pad = win // 2
            kernel = np.ones((win,), dtype=np.float32) / float(win)
            T, N, D = vecs.shape
            v = np.empty_like(vecs, dtype=np.float32)
            padded = np.pad(vecs, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            for n in range(N):
                for d in range(D):
                    v[:, n, d] = np.convolve(padded[:, n, d], kernel, mode="valid")
        norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
        return v / norm
    # Trajectory
    T, N = 120, 16
    steps = 0.05 * np.random.randn(T, N, 3)
    pos = steps.cumsum(axis=0)

    # Orientation as direction vectors: finite difference along time, then smooth
    dirs = np.zeros_like(pos, dtype=np.float32)
    dirs[1:] = pos[1:] - pos[:-1]
    dirs[0] = dirs[1]
    dirs = smooth_vectors_time(dirs, win=7)
    ori = dirs  # (T, N, 3) direction vectors

    # Per-frame node colors: oscillate between red and green
    t = np.linspace(0, 2 * np.pi, T)
    w = (np.sin(t) * 0.5 + 0.5)[:, None, None]  # (T,1,1)
    red = np.broadcast_to(np.array([1.0, 0.1, 0.1])[None, None, :], (T, N, 3))
    green = np.broadcast_to(np.array([0.1, 1.0, 0.1])[None, None, :], (T, N, 3))
    node_colors = w * red + (1.0 - w) * green

    # Per-frame sizes: pulsate
    # Interpret sizes as multipliers around 1.0 to ensure visibility
    node_sizes = 1.0 + 0.3 * ((np.sin(t)[:, None] * 0.5 + 0.5) - 0.5)

    # Graph edges (path graph)
    G = nx.path_graph(N)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "blender_3d_cones_graph_demo.mp4")

    cfg = {
        "backend": "blender",
        "fps": 24,
        "save_path": out_path,
        # Nodes as cones
        "primitive_shape": "cone",
        # Base cone scale to pair with node_sizes multiplier
        "obj_scales": 0.2,
        # Make cones vivid green as a default
        "primitive_color": [0.15, 0.9, 0.2],
        # Edge thickness
        "edge_bevel": 0.015,
        "save_blend": True,
        # If Blender isn't discovered automatically on macOS, set blender_exec:
        # "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender",
    }

    visualize_dynamics(
        cfg,
        pos,
        ori=ori,  # (T,N,3) direction vectors
        graph=G,
        node_colors=node_colors,
        node_sizes=node_sizes,
    )
    print(f"Rendered Blender cones+graph demo to: {out_path}")


if __name__ == "__main__":
    main()

