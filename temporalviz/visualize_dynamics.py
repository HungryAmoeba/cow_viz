from .base import create_visualizer
import numpy as np
from typing import Any, Dict, Optional, Union


def visualize_dynamics(
    config: Union[str, Dict[str, Any]],
    pos: np.ndarray,
    ori: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    **kwargs: Any,
):
    """
    Visualize the dynamics of particles using the selected backend.

    Parameters
    ----------
    - config: dict|str|VisualizerConfig selecting backend and options
    - pos: (T, N, D) positions over time
    - ori: optional (T, N, D) or (T, N, 4) orientations over time
    - save_path: optional output path for saved animation/render
    - **kwargs: backend-specific options (e.g., graph, node_colors, node_sizes, edge_colors, etc.)
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    # Basic validation
    if not isinstance(pos, np.ndarray):
        try:
            pos = np.array(pos)
        except Exception:
            raise TypeError("pos must be array-like of shape (T, N, D)")
    if pos.ndim != 3:
        raise ValueError(f"pos must have shape (T, N, D), got {pos.shape}")
    D = pos.shape[-1]
    if D not in (2, 3):
        raise ValueError(f"pos last dimension D must be 2 or 3, got {D}")
    if ori is not None:
        if not isinstance(ori, np.ndarray):
            try:
                ori = np.array(ori)
            except Exception:
                raise TypeError("ori must be array-like of shape (T, N, D) or (T, N, 4)")
        if ori.ndim != 3:
            raise ValueError(f"ori must have shape (T, N, D) or (T, N, 4), got {ori.shape}")
        if ori.shape[0] != pos.shape[0] or ori.shape[1] != pos.shape[1]:
            raise ValueError(f"ori leading dims must match pos (T,N,·). Got ori {ori.shape}, pos {pos.shape}")
        if ori.shape[-1] not in (D, 4, 3):
            raise ValueError(f"ori last dim must be {D}, 3, or 4; got {ori.shape[-1]}")

    visualizer = create_visualizer(config)
    return visualizer.visualize(pos, ori=ori, save_path=save_path, **kwargs)