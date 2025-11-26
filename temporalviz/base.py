"""Base visualization interfaces and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union


class BaseVisualizer:
    """
    Abstract base class for all visualizers.

    Subclasses must implement visualize().
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def visualize(self, pos, ori=None, save_path: Optional[str] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement visualize()")


@dataclass
class VisualizerConfig:
    """Lightweight config object for selecting and configuring a backend.

    - backend: "matplotlib" or "blender"
    - any additional keyword args are forwarded to the specific backend
    """

    backend: str = "matplotlib"
    # Additional attributes are allowed dynamically via dict-like usage

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _normalize_config(config: Union[str, Dict[str, Any], VisualizerConfig, None]) -> Dict[str, Any]:
    if config is None:
        return {"backend": "matplotlib"}
    if isinstance(config, str):
        return {"backend": config}
    if isinstance(config, VisualizerConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return {**config}
    raise TypeError("config must be str | dict | VisualizerConfig | None")


def create_visualizer(config: Union[str, Dict[str, Any], VisualizerConfig, None] = None) -> BaseVisualizer:
    """
    Create a visualizer instance based on a simple config.

    Examples
    --------
    - create_visualizer("matplotlib")
    - create_visualizer({"backend": "matplotlib", "interval": 40})
    - create_visualizer(VisualizerConfig(backend="blender"))
    """
    cfg = _normalize_config(config)
    backend = cfg.get("backend", "matplotlib").lower()

    if backend == "matplotlib":
        from .matplotlib_visualization import MatplotlibVisualizer

        return MatplotlibVisualizer(cfg)
    if backend == "blender":
        from .blender_visualization import BlenderVisualizer

        return BlenderVisualizer(cfg)

    raise ValueError(f"Unknown backend '{backend}'. Supported: 'matplotlib', 'blender'.")
    