import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional


def _compute_bounds(pos):
    all_pos = pos.reshape(-1, pos.shape[-1])
    min_vals = all_pos.min(axis=0)
    max_vals = all_pos.max(axis=0)
    padding = 0.1 * (max_vals - min_vals + 1e-9)
    return min_vals, max_vals, padding


def _initial_sizes(node_sizes):
    if node_sizes is None:
        return 100
    if node_sizes.ndim == 2:
        return node_sizes[0]
    return node_sizes


def _initial_colors(node_colors):
    if node_colors is None:
        return None
    if hasattr(node_colors, "ndim") and node_colors.ndim >= 2:
        return node_colors[0]
    return node_colors


def _auto_edge_alpha_scale(pos):
    # Use median pairwise distance at t=0 as a scale
    p0 = pos[0]
    diffs = p0[:, None, :] - p0[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    d = d[np.triu_indices_from(d, k=1)]
    if d.size == 0:
        return 1.0
    return float(np.median(d) + 1e-9)


def animate_temporal_graph(
    pos: np.ndarray,
    graph: nx.Graph,
    ori: Optional[np.ndarray] = None,
    node_colors=None,
    edge_colors=None,
    node_sizes=None,
    interval: int = 200,
    title: Optional[str] = None,
    draw_edges: bool = True,
    edge_opacity_by_distance: bool = False,
    edge_opacity_scale: Optional[float] = None,
    edge_width: float = 1.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    **kwargs,
):
    """
    Animate a temporal graph with node positions and optional orientations.

    - pos: (T, N, D) D in {2,3}
    - graph: networkx graph with N nodes
    - ori: (T, N, D) or (T, N) optional
    - node_colors: (N,), (T,N), or (T,N,3/4)
    - node_sizes: (N,) or (T,N)
    - draw_edges: draw edges if True
    - edge_opacity_by_distance: if True, alpha decays with edge length
    - edge_opacity_scale: distance scale for alpha, default median pairwise distance at t=0
    """
    T, N, D = pos.shape
    if D not in (2, 3):
        raise ValueError("Only 2D or 3D supported")

    # Prepare figure/axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d" if D == 3 else None)

    # Bounds and labels
    mn, mx, pad = _compute_bounds(pos)
    if D == 3:
        ax.set_xlim3d(mn[0] - pad[0], mx[0] + pad[0])
        ax.set_ylim3d(mn[1] - pad[1], mx[1] + pad[1])
        ax.set_zlim3d(mn[2] - pad[2], mx[2] + pad[2])
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
    else:
        ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
        ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # Node scatter
    initial_sizes = _initial_sizes(node_sizes)
    initial_colors = _initial_colors(node_colors)
    if D == 2:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], c=initial_colors, s=initial_sizes, zorder=2)
    else:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], c=initial_colors, s=initial_sizes)

    # Edge artists
    edge_lines = []
    if draw_edges and graph.number_of_edges() > 0:
        for i, j in graph.edges:
            if D == 2:
                line, = ax.plot(
                    [pos[0, i, 0], pos[0, j, 0]],
                    [pos[0, i, 1], pos[0, j, 1]],
                    c=(edge_colors if isinstance(edge_colors, str) else "gray"),
                    lw=edge_width,
                    alpha=0.6,
                    zorder=1,
                )
            else:
                line, = ax.plot(
                    [pos[0, i, 0], pos[0, j, 0]],
                    [pos[0, i, 1], pos[0, j, 1]],
                    [pos[0, i, 2], pos[0, j, 2]],
                    c=(edge_colors if isinstance(edge_colors, str) else "gray"),
                    lw=edge_width,
                    alpha=0.6,
                )
            edge_lines.append(line)

    # Orientation arrows (2D quiver; optional 3D quiver)
    quivers = None
    if ori is not None:
        p0 = pos[0]
        if D == 2:
            # scale quivers to half the min nonzero distance at t=0
            diffs = p0[:, None, :] - p0[None, :, :]
            d2 = np.sum(diffs**2, axis=-1)
            d2[np.eye(N, dtype=bool)] = np.inf
            arrow_len = 0.5 * float(np.sqrt(np.min(d2))) if np.isfinite(d2).any() else 0.1
            quivers = ax.quiver(
                p0[:, 0], p0[:, 1],
                ori[0, :, 0], ori[0, :, 1],
                color="red", scale_units="xy", scale=1.0 / max(arrow_len, 1e-9), zorder=1,
            )
        else:
            # 3D quiver: choose length proportional to cloud size
            span = float(np.max(np.linalg.norm(p0 - p0.mean(axis=0), axis=1) + 1e-9))
            qlen = 0.3 * (span if np.isfinite(span) and span > 0 else 1.0)
            quivers = ax.quiver(
                p0[:, 0], p0[:, 1], p0[:, 2],
                ori[0, :, 0], ori[0, :, 1], ori[0, :, 2],
                length=qlen, normalize=True, color="red",
            )

    # Edge opacity setup
    if edge_opacity_by_distance:
        if edge_opacity_scale is None:
            edge_opacity_scale = _auto_edge_alpha_scale(pos)
        edge_opacity_scale = float(edge_opacity_scale)

    def _update_edges(frame):
        if not edge_lines:
            return
        for (i_e, (i, j)) in enumerate(graph.edges):
            line = edge_lines[i_e]
            if D == 2:
                line.set_data([pos[frame, i, 0], pos[frame, j, 0]], [pos[frame, i, 1], pos[frame, j, 1]])
            else:
                line.set_data_3d(
                    [pos[frame, i, 0], pos[frame, j, 0]],
                    [pos[frame, i, 1], pos[frame, j, 1]],
                    [pos[frame, i, 2], pos[frame, j, 2]],
                )
            if edge_opacity_by_distance:
                d = np.linalg.norm(pos[frame, i] - pos[frame, j])
                alpha = np.exp(-d / max(edge_opacity_scale, 1e-9))
                line.set_alpha(alpha)
            if edge_colors is not None and not isinstance(edge_colors, str):
                # If a list/array of per-edge colors is provided, map by index
                try:
                    line.set_color(edge_colors[i_e])
                except Exception:
                    pass

    def update(frame):
        nonlocal quivers
        # positions
        if D == 2:
            scat.set_offsets(pos[frame])
        else:
            scat._offsets3d = (pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2])

        # node colors
        if node_colors is not None:
            if hasattr(node_colors, "ndim") and node_colors.ndim >= 2:
                scat.set_color(node_colors[frame])
            else:
                scat.set_color(node_colors)

        # node sizes
        if node_sizes is not None:
            if node_sizes.ndim == 2:
                scat.set_sizes(node_sizes[frame])
            else:
                scat.set_sizes(node_sizes)

        # edges
        _update_edges(frame)

        # orientations
        if ori is not None and quivers is not None:
            if D == 2:
                quivers.set_offsets(pos[frame])
                quivers.set_UVC(ori[frame, :, 0], ori[frame, :, 1])
            else:
                # Recreate 3D quiver each frame (mpl 3D quiver lacks simple setters)
                try:
                    quivers.remove()
                except Exception:
                    pass
                span = float(np.max(np.linalg.norm(pos[frame] - pos[frame].mean(axis=0), axis=1) + 1e-9))
                qlen = 0.3 * (span if np.isfinite(span) and span > 0 else 1.0)
                quivers = ax.quiver(
                    pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2],
                    ori[frame, :, 0], ori[frame, :, 1], ori[frame, :, 2],
                    length=qlen, normalize=True, color="red",
                )

        ax.set_title(f"{title or 'Temporal Graph'}\nFrame {frame+1}/{T}")
        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.close()
    return ani
    