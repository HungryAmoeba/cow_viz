import os
import numpy as np

from temporalviz.visualize_dynamics import visualize_dynamics


def quat_from_axis_angle(axis: str, angle: float) -> np.ndarray:
    """Return (w, x, y, z) quaternion for rotation 'angle' about canonical axis."""
    half = 0.5 * angle
    c = float(np.cos(half))
    s = float(np.sin(half))
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=float)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=float)
    if axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=float)
    raise ValueError("axis must be one of: 'x', 'y', 'z'")


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q = q1 * q2, both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def main():
    # Output location
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "blender_3d_quaternion_demo.mp4")

    # Simple 3D motion: two particles on counter-rotating circles in the XY plane
    T = 180
    N = 2
    t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
    r = 2.0

    pos = np.zeros((T, N, 3), dtype=float)
    # Particle 0: CCW circle
    pos[:, 0, 0] = r * np.cos(t)
    pos[:, 0, 1] = r * np.sin(t)
    pos[:, 0, 2] = 0.0
    # Particle 1: CW circle
    pos[:, 1, 0] = r * np.cos(-t)
    pos[:, 1, 1] = r * np.sin(-t)
    pos[:, 1, 2] = 0.0

    # Build quaternions (w, x, y, z) explicitly.
    # We'll orient cones to lie in the XY plane and point along the tangent direction.
    # Blender cones point along +Z by default, so we first rotate -90° about X to align cone axis with +Y,
    # then yaw around Z so +Y matches the instantaneous tangent direction.
    ori = np.zeros((T, N, 4), dtype=float)
    tilt = quat_from_axis_angle("x", -0.5 * np.pi)  # rotate cone axis from +Z to +Y

    # Particle 0 yaw (CCW): tangent angle is theta + pi/2
    yaw0 = t + 0.5 * np.pi
    # Particle 1 yaw (CW): tangent angle is -theta - pi/2
    yaw1 = -t - 0.5 * np.pi

    for i in range(T):
        q0 = quat_multiply(quat_from_axis_angle("z", float(yaw0[i])), tilt)
        q1 = quat_multiply(quat_from_axis_angle("z", float(yaw1[i])), tilt)
        ori[i, 0] = q0 / np.linalg.norm(q0)
        ori[i, 1] = q1 / np.linalg.norm(q1)

    # Render with Blender backend using cones to make orientation obvious
    config = {
        "backend": "blender",
        "fps": 24,
        "primitive_shape": "cone",
        "primitive_color": (0.15, 0.8, 0.3),
        # Slightly larger cones for visibility
        "obj_scales": 0.25,
    }

    visualize_dynamics(config, pos, ori=ori, save_path=save_path)
    print(f"Wrote: {save_path}")


if __name__ == "__main__":
    main()

