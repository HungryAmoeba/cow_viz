import numpy as np
import tempfile
import subprocess
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, Tuple, Iterable

from .base import BaseVisualizer


class BlenderVisualizer(BaseVisualizer):
    """
    Visualizer for particle dynamics using Blender.

    Config keys (optional):
    - blender_exec: path to Blender executable (auto-detected if missing)
    - obj_paths: list[str|None] per-agent .obj paths, or None to use primitives
    - obj_scales: float or list[float] per-agent uniform scales
    - fps: int frames per second for Blender timeline (default 24)
    - save_path: path where Blender should render/save (optional)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def visualize(
        self,
        pos,
        ori=None,
        save_path: Optional[str] = None,
        graph=None,
        node_colors=None,
        node_sizes=None,
        edge_colors=None,
        **kwargs,
    ):
        return animate_particle_motion_blender(
            pos,
            ori,
            save_path=save_path or self.config.get("save_path"),
            obj_paths=self.config.get("obj_paths"),
            obj_scales=self.config.get("obj_scales"),
            fps=self.config.get("fps", 24),
            blender_exec=self.config.get("blender_exec"),
            primitive_shape=self.config.get("primitive_shape"),
            primitive_color=self.config.get("primitive_color"),
            save_blend=self.config.get("save_blend"),
            blend_path=self.config.get("blend_path"),
            graph=graph,
            node_colors=node_colors,
            node_sizes=node_sizes,
            edge_colors=edge_colors,
            edge_bevel=self.config.get("edge_bevel"),
        )


def animate_particle_motion_blender(
    pos: np.ndarray,
    ori: Optional[np.ndarray],
    save_path: Optional[str] = None,
    obj_paths: Optional[Sequence[Optional[str]]] = None,
    obj_scales: Optional[Sequence[float]] = None,
    fps: int = 24,
    blender_exec: Optional[str] = None,
    primitive_shape: Optional[str] = None,
    primitive_color: Optional[Sequence[float]] = None,
    save_blend: Optional[bool] = None,
    blend_path: Optional[str] = None,
    graph: Any = None,
    node_colors: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    edge_colors: Optional[np.ndarray] = None,
    edge_bevel: Optional[float] = None,
    # Cone shape parameters (optional)
    cone_radius1: Optional[float] = None,
    cone_radius2: Optional[float] = None,
    cone_depth: Optional[float] = None,
) -> None:
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    # Normalize orientation
    if ori is None:
        T, N, _ = pos.shape
        ori = np.zeros((T, N, 4), dtype=float)
        # Identity quaternion in (w, x, y, z) format
        ori[:, :, 0] = 1.0
    # Else: leave as provided (support (T,N,3) direction or (T,N,4) quaternion)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        data_path = tmpdir_p / "data.npz"
        opts_path = tmpdir_p / "options.json"

        arrays: Dict[str, Any] = {"pos": pos, "ori": ori}
        # Optional per-frame arrays
        if node_colors is not None:
            arrays["node_colors"] = np.array(node_colors)
        if node_sizes is not None:
            arrays["node_sizes"] = np.array(node_sizes)
        if edge_colors is not None:
            arrays["edge_colors"] = np.array(edge_colors)
        # Graph edges
        edges_arr = None
        if graph is not None:
            try:
                import networkx as nx  # type: ignore

                if isinstance(graph, nx.Graph):
                    edges_arr = np.array(list(graph.edges()), dtype=int)
                else:
                    edges_arr = np.array(graph, dtype=int)
            except Exception:
                # Attempt generic conversion
                try:
                    edges_arr = np.array(graph, dtype=int)
                except Exception:
                    edges_arr = None
        if edges_arr is not None:
            arrays["edges"] = edges_arr

        np.savez_compressed(data_path, **arrays)

        options: Dict[str, Any] = {
            "save_path": save_path,
            "fps": int(fps),
        }
        if save_blend is not None:
            options["save_blend"] = bool(save_blend)
        if blend_path is not None:
            options["blend_path"] = str(blend_path)
        if obj_paths is not None:
            options["obj_paths"] = list(obj_paths)
        if obj_scales is not None:
            # Broadcast single float to list of length N if needed
            if isinstance(obj_scales, (int, float)):
                options["obj_scales"] = float(obj_scales)
            else:
                options["obj_scales"] = list(obj_scales)
        if primitive_shape is not None:
            options["primitive_shape"] = str(primitive_shape)
        if primitive_color is not None:
            try:
                r, g, b = primitive_color[:3]
                options["primitive_color"] = [float(r), float(g), float(b)]
            except Exception:
                pass
        if edge_bevel is not None:
            try:
                options["edge_bevel"] = float(edge_bevel)
            except Exception:
                pass
        # Cone parameter passthrough
        if cone_radius1 is not None:
            try:
                options["cone_radius1"] = float(cone_radius1)
            except Exception:
                pass
        if cone_radius2 is not None:
            try:
                options["cone_radius2"] = float(cone_radius2)
            except Exception:
                pass
        if cone_depth is not None:
            try:
                options["cone_depth"] = float(cone_depth)
            except Exception:
                pass

        opts_path.write_text(json.dumps(options))

        blender_script = str(Path(__file__).parent / "blender_script.py")
        blender_exec = blender_exec or find_blender_executable()
        if not Path(blender_exec).exists():
            raise FileNotFoundError(
                f"Blender executable not found at '{blender_exec}'. "
                "Install Blender and/or pass config {'blender_exec': '/path/to/Blender'}."
            )

        cmd = [
            blender_exec,
            "--background",
            "--python",
            blender_script,
            "--",
            str(data_path),
            str(opts_path),
        ]

        subprocess.run(cmd, check=True)


def find_blender_executable() -> str:
    """
    Best-effort guess for Blender executable. Does not guarantee existence.
    """
    if sys.platform.startswith("linux"):
        return "/usr/bin/blender"
    if sys.platform == "darwin":
        # Default app bundle path
        return "/Applications/Blender.app/Contents/MacOS/Blender"
    if sys.platform == "win32":
        return "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe"
    raise OSError("Unsupported OS for Blender execution")
    