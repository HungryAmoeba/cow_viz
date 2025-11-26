import argparse
import sys
from pathlib import Path

import numpy as np

from .visualize_dynamics import visualize_dynamics


def _example_brownian_matplotlib(save: str | None = None) -> None:
    T, N = 200, 32
    steps = 0.12 * np.random.randn(T, N, 2)
    pos = steps.cumsum(axis=0)
    cfg = {"backend": "matplotlib", "interval": 30, "title": "Brownian Motion (2D)"}
    visualize_dynamics(cfg, pos, save_path=save)


def _example_graph_matplotlib(save: str | None = None) -> None:
    import networkx as nx

    T, N = 120, 24
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 4 * np.pi, T)
    pos = np.zeros((T, N, 2))
    pos[:, :, 0] = x[None, :]
    pos[:, :, 1] = 0.25 * np.sin(t[:, None] + 2 * np.pi * x[None, :])

    G = nx.path_graph(N)
    cfg = {
        "backend": "matplotlib",
        "interval": 40,
        "title": "Path Graph Wiggle",
        "edge_width": 1.5,
    }
    visualize_dynamics(cfg, pos, save_path=save, graph=G)


def _example_random_walk_blender(save: str | None = None) -> None:
    # Simple 3D random walk with cones oriented along velocity
    T, N = 120, 4
    steps = 0.08 * np.random.randn(T, N, 3)
    pos = steps.cumsum(axis=0)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    cfg = {
        "backend": "blender",
        "fps": 24,
        "primitive_shape": "cone",
        "primitive_color": [0.2, 0.7, 0.2],
        "edge_bevel": 0.015,
        "save_path": save,
    }
    visualize_dynamics(cfg, pos, ori=vel, save_path=save)


def run_examples(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="temporalviz-examples",
        description="Run TemporalViz examples (Matplotlib by default; Blender optional).",
    )
    parser.add_argument(
        "example",
        nargs="?",
        default="brownian",
        choices=["brownian", "graph", "blender-walk"],
        help="Which example to run.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save output (e.g., out.mp4).",
    )
    args = parser.parse_args(argv)

    save_path = args.save
    if save_path is not None:
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.example == "brownian":
        _example_brownian_matplotlib(save_path)
    elif args.example == "graph":
        _example_graph_matplotlib(save_path)
    elif args.example == "blender-walk":
        _example_random_walk_blender(save_path)
    else:
        parser.error(f"Unknown example '{args.example}'")


if __name__ == "__main__":
    run_examples(sys.argv[1:])

