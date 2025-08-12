from .base import create_visualizer
import numpy as np


def visualize_dynamics(config, pos, ori=None, save_path=None):
    """
    Visualize the dynamics of particles using the selected backend.

    Parameters
    ----------
    - config: dict|str|VisualizerConfig selecting backend and options
    - pos: (T, N, D) positions over time
    - ori: optional (T, N, D) or (T, N, 4) orientations over time
    - save_path: optional output path for saved animation/render
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    visualizer = create_visualizer(config)
    return visualizer.visualize(pos, ori=ori, save_path=save_path)