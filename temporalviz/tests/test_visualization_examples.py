import os
import numpy as np
import networkx as nx

import matplotlib


def _force_headless_matplotlib():
    # Ensure tests do not try to open GUI windows
    backend = os.environ.get("MPLBACKEND", "Agg")
    matplotlib.use(backend)


def test_brownian_motion_matplotlib_headless():
    _force_headless_matplotlib()
    from temporalviz import visualize_dynamics

    T, N = 40, 8
    steps = 0.1 * np.random.randn(T, N, 2)
    pos = steps.cumsum(axis=0)

    anim = visualize_dynamics({"backend": "matplotlib", "interval": 10}, pos)
    # Basic sanity checks on returned animation object
    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)


def test_path_graph_wiggle_edges_render():
    _force_headless_matplotlib()
    from temporalviz import create_visualizer

    T, N = 30, 12
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 2 * np.pi, T)
    pos = np.zeros((T, N, 2))
    pos[:, :, 0] = x[None, :]
    pos[:, :, 1] = 0.2 * np.sin(t[:, None] + 2 * np.pi * x[None, :])

    G = nx.path_graph(N)
    viz = create_visualizer({"backend": "matplotlib", "interval": 10})
    anim = viz.visualize(pos, graph=G)

    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)

