import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
from typing import Any, Dict, Optional

from .temporal_graph_matplotlib import animate_temporal_graph
from .base import BaseVisualizer


def to_numpy(arr):
    if arr is not None and not isinstance(arr, np.ndarray):
        try:
            return np.array(arr)
        except Exception:
            pass
    return arr


def rescale_node_sizes(
    sizes,
    min_size: float = 50.0,
    max_size: float = 500.0,
    shift: bool = True,
):
    sizes = np.asarray(sizes)
    if sizes.ndim == 1:
        sizes = sizes[None, :]  # (1, N) for static
    min_val = np.min(sizes)
    max_val = np.max(sizes)
    if shift:
        sizes = sizes - min_val
        max_val = np.max(sizes)
        min_val = 0
    if max_val > min_val:
        scaled = min_size + (sizes - min_val) * (max_size - min_size) / (max_val - min_val)
    else:
        scaled = np.full_like(sizes, (min_size + max_size) / 2)
    return scaled.squeeze()


class MatplotlibVisualizer(BaseVisualizer):
    """
    Visualizer for particle dynamics and temporal graphs using Matplotlib.

    Config keys (all optional):
    - interval: int ms between frames (default 50)
    - title: str window title
    - draw_edges: bool whether to draw edges (default True if graph provided)
    - auto_size: bool rescale provided sizes to a nice visual range (default True)
    - node_size_range: (min,max) range for auto sizing (default (50, 500))
    - edge_opacity_by_distance: bool; if True, edge alpha decays with distance (default False)
    - edge_opacity_scale: float distance scale for alpha decay (default auto from data)
    - edge_width: float matplotlib line width (default 1.5)
    - xlabel, ylabel, zlabel: axis labels
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def visualize(
        self,
        pos,
        ori=None,
        save_path: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        node_colors=None,
        edge_colors=None,
        node_sizes=None,
        **kwargs,
    ):
        # Robust conversions
        pos = to_numpy(pos)
        ori = to_numpy(ori)
        node_colors = to_numpy(node_colors)
        edge_colors = to_numpy(edge_colors)
        node_sizes = to_numpy(node_sizes)

        interval = self.config.get("interval", 50)
        title = self.config.get("title")
        draw_edges = self.config.get("draw_edges")
        auto_size = self.config.get("auto_size", True)
        size_range = self.config.get("node_size_range", (50, 500))

        # Allow overrides via kwargs
        interval = kwargs.pop("interval", interval)
        title = kwargs.pop("title", title)
        if "draw_edges" in kwargs:
            draw_edges = kwargs.pop("draw_edges")

        if graph is None:
            # Create a graph with N nodes and no edges
            N = pos.shape[1]
            graph = nx.empty_graph(N)
            if draw_edges is None:
                draw_edges = False
        else:
            if draw_edges is None:
                draw_edges = True

        # Shape checks
        if pos.ndim != 3:
            raise ValueError(f"pos must be (T, N, D), got shape {pos.shape}")
        if ori is not None and (ori.ndim not in (2, 3)):
            raise ValueError(f"ori must be (T, N, D) or (T, N), got shape {ori.shape}")

        # Auto-scale node sizes if provided
        if node_sizes is not None and auto_size:
            node_sizes = rescale_node_sizes(node_sizes, min_size=size_range[0], max_size=size_range[1])

        anim = animate_temporal_graph(
            pos,
            graph,
            ori=ori,
            node_colors=node_colors,
            edge_colors=edge_colors,
            node_sizes=node_sizes,
            interval=interval,
            title=title,
            draw_edges=draw_edges,
            edge_opacity_by_distance=self.config.get("edge_opacity_by_distance", False),
            edge_opacity_scale=self.config.get("edge_opacity_scale"),
            edge_width=self.config.get("edge_width", 1.5),
            xlabel=self.config.get("xlabel"),
            ylabel=self.config.get("ylabel"),
            zlabel=self.config.get("zlabel"),
            **kwargs,
        )

        if save_path is not None:
            fps = int(max(1, 1000 // int(interval)))
            anim.save(save_path, writer="ffmpeg", fps=fps, extra_args=["-vcodec", "libx264"])
        return anim
        