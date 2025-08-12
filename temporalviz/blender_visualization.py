import numpy as np
import tempfile
import subprocess
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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

    def visualize(self, pos, ori=None, save_path: Optional[str] = None, **kwargs):
        return animate_particle_motion_blender(
            pos,
            ori,
            save_path=save_path or self.config.get("save_path"),
            obj_paths=self.config.get("obj_paths"),
            obj_scales=self.config.get("obj_scales"),
            fps=self.config.get("fps", 24),
            blender_exec=self.config.get("blender_exec"),
        )


def animate_particle_motion_blender(
    pos: np.ndarray,
    ori: Optional[np.ndarray],
    save_path: Optional[str] = None,
    obj_paths: Optional[Sequence[Optional[str]]] = None,
    obj_scales: Optional[Sequence[float]] = None,
    fps: int = 24,
    blender_exec: Optional[str] = None,
):
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    # Normalize orientation
    if ori is None:
        T, N, _ = pos.shape
        ori = np.zeros((T, N, 4), dtype=float)
        ori[:, :, -1] = 1.0
    elif ori.shape[-1] == 3:
        # Convert direction vectors to quaternion-like [0, x, y, z]
        z = np.zeros(ori.shape[:-1] + (1,), dtype=ori.dtype)
        ori = np.concatenate([z, ori], axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        data_path = tmpdir_p / "data.npz"
        opts_path = tmpdir_p / "options.json"

        np.savez_compressed(data_path, pos=pos, ori=ori)

        options: Dict[str, Any] = {
            "save_path": save_path,
            "fps": int(fps),
        }
        if obj_paths is not None:
            options["obj_paths"] = list(obj_paths)
        if obj_scales is not None:
            # Broadcast single float to list of length N if needed
            if isinstance(obj_scales, (int, float)):
                options["obj_scales"] = float(obj_scales)
            else:
                options["obj_scales"] = list(obj_scales)

        opts_path.write_text(json.dumps(options))

        blender_script = str(Path(__file__).parent / "blender_script.py")
        blender_exec = blender_exec or find_blender_executable()

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
    if sys.platform.startswith("linux"):
        return "/usr/bin/blender"
    if sys.platform == "darwin":
        return "/Applications/Blender.app/Contents/MacOS/Blender"
    if sys.platform == "win32":
        return "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe"
    raise OSError("Unsupported OS for Blender execution")