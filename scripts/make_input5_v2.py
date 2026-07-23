from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINT_IDS_MAP = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

# Unitree SDK joint order: left leg, right leg, waist, left arm, right arm.
STAND_JOINT_POS = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def smoothstep5(t: np.ndarray) -> np.ndarray:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = normalize_quat(q0.astype(np.float64))
    q1 = normalize_quat(q1.astype(np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return normalize_quat(q0 + alpha[:, None] * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - alpha) * theta)[:, None] / sin_theta * q0
        + np.sin(alpha * theta)[:, None] / sin_theta * q1
    )


def yaw_only(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(quat_wxyz)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def blend_segment(
    root_pos_a: np.ndarray,
    root_pos_b: np.ndarray,
    root_quat_a: np.ndarray,
    root_quat_b: np.ndarray,
    joint_a: np.ndarray,
    joint_b: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = smoothstep5(np.linspace(0.0, 1.0, frames + 1))
    root_pos = root_pos_a + alpha[:, None] * (root_pos_b - root_pos_a)
    root_quat = quat_slerp(root_quat_a, root_quat_b, alpha)
    joint_pos = joint_a + alpha[:, None] * (joint_b - joint_a)
    return root_pos, root_quat, joint_pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--stand-height", type=float, default=0.76)
    args = parser.parse_args()

    motion = np.load(args.input)
    fps = int(motion["fps"][0])
    transition_frames = round(args.transition_seconds * fps)
    hold_frames = round(args.hold_seconds * fps)

    root_pos = motion["body_pos_w"][:, 0, :].astype(np.float64)
    root_quat = normalize_quat(motion["body_quat_w"][:, 0, :].astype(np.float64))
    joint_bfs = motion["joint_pos"].astype(np.float64)
    joint_sdk = np.empty_like(joint_bfs)
    joint_sdk[:, JOINT_IDS_MAP] = joint_bfs

    start_stand_pos = np.array([root_pos[0, 0], root_pos[0, 1], args.stand_height])
    end_stand_pos = np.array([root_pos[-1, 0], root_pos[-1, 1], args.stand_height])
    start_stand_quat = yaw_only(root_quat[0])
    end_stand_quat = yaw_only(root_quat[-1])

    start = blend_segment(
        start_stand_pos, root_pos[0], start_stand_quat, root_quat[0],
        STAND_JOINT_POS, joint_sdk[0], transition_frames,
    )
    end = blend_segment(
        root_pos[-1], end_stand_pos, root_quat[-1], end_stand_quat,
        joint_sdk[-1], STAND_JOINT_POS, transition_frames,
    )

    out_root_pos = np.concatenate(
        [start[0][:-1], root_pos, end[0][1:], np.repeat(end_stand_pos[None], hold_frames, axis=0)]
    )
    out_root_quat = np.concatenate(
        [start[1][:-1], root_quat, end[1][1:], np.repeat(end_stand_quat[None], hold_frames, axis=0)]
    )
    out_joint = np.concatenate(
        [start[2][:-1], joint_sdk, end[2][1:], np.repeat(STAND_JOINT_POS[None], hold_frames, axis=0)]
    )
    out = np.concatenate([out_root_pos, out_root_quat[:, [1, 2, 3, 0]], out_joint], axis=1)

    if not np.isfinite(out).all() or out.shape[1] != 36:
        raise RuntimeError(f"invalid output shape/data: {out.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, out, delimiter=",", fmt="%.9g")
    print(f"wrote {args.output}")
    print(f"fps={fps}, frames={len(out)}, duration={(len(out) - 1) / fps:.2f}s")


if __name__ == "__main__":
    main()
