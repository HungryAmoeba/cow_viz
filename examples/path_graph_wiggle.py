import numpy as np
import networkx as nx

from temporalviz import create_visualizer

_anim = None  # keep a global ref so it isn't garbage-collected

def main():
    import os

    T, N = 240, 32
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 6 * np.pi, T)
    pos = np.zeros((T, N, 2))
    pos[:, :, 0] = x[None, :]
    pos[:, :, 1] = 0.25 * np.sin(t[:, None] + 2 * np.pi * x[None, :])

    G = nx.path_graph(N)
    viz = create_visualizer({"backend": "matplotlib", "interval": 30, "title": "Path Graph Wiggle"})
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "path_graph_wiggle.mp4")
    global _anim
    _anim = viz.visualize(pos, graph=G, save_path=out_path)
    print(f"Saved animation to: {out_path}")


if __name__ == "__main__":
    main()

