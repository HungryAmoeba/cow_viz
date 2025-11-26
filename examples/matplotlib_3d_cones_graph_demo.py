import os
import numpy as np
import networkx as nx

from temporalviz import visualize_dynamics


def smooth_vectors_time(vecs: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Simple moving-average smoothing along time for (T, N, D) arrays.
    Window size `win` should be odd. Result is re-normalized per vector.
    """
    if win <= 1:
        # no smoothing
        v = vecs.copy()
    else:
        pad = win // 2
        kernel = np.ones((win,), dtype=np.float32) / float(win)
        T, N, D = vecs.shape
        v = np.empty_like(vecs, dtype=np.float32)
        # pad along time with edge values
        padded = np.pad(vecs, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        # convolve per (N,D) along time
        for n in range(N):
            for d in range(D):
                v[:, n, d] = np.convolve(padded[:, n, d], kernel, mode="valid")
    # renormalize to unit vectors (avoid division by zero)
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm


def main():
    # 3D random walk with smooth directions
    T, N = 150, 24
    rng = np.random.default_rng(0)
    steps = 0.05 * rng.standard_normal((T, N, 3), dtype=np.float32)
    pos = np.cumsum(steps, axis=0)

    # Direction estimates from finite differences, then smoothed
    dirs = np.zeros_like(pos)
    dirs[1:] = pos[1:] - pos[:-1]
    dirs[0] = dirs[1]
    dirs = smooth_vectors_time(dirs, win=7)  # smoother orientation changes

    # Colors: fade between two colors over time
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    w = (np.sin(t) * 0.5 + 0.5).astype(np.float32)[:, None, None]
    c1 = np.array([0.15, 0.9, 0.2], dtype=np.float32)  # green
    c2 = np.array([0.2, 0.4, 1.0], dtype=np.float32)   # blue
    node_colors = w * c1 + (1.0 - w) * c2
    node_colors = np.broadcast_to(node_colors, (T, N, 3)).copy()

    # Sizes: mild pulsation
    node_sizes = 100.0 * (0.8 + 0.4 * (np.sin(t)[:, None] * 0.5 + 0.5))

    # 3D path graph edges
    G = nx.path_graph(N)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matplotlib_3d_cones_graph_demo.mp4")

    cfg = {
        "backend": "matplotlib",
        "interval": 40,
        "title": "3D Graph with Direction Vectors (Matplotlib)",
        "draw_edges": True,                 # ensure edges are drawn
        "edge_opacity_by_distance": False,  # keep edges fully visible
        "edge_width": 3.5,
        # Give edges a strong, visible color
        "edge_colors": "black",
        "xlabel": "X", "ylabel": "Y", "zlabel": "Z",
        # set auto_size False to use sizes directly (already scaled)
        "auto_size": False,
    }

    visualize_dynamics(
        cfg,
        pos,
        ori=dirs,               # 3D direction vectors
        graph=G,
        node_colors=node_colors,
        node_sizes=node_sizes,
        save_path=out_path,
    )
    print(f"Saved Matplotlib 3D graph demo to: {out_path}")


if __name__ == "__main__":
    main()

