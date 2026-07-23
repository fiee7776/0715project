from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        raise ValueError("Quaternion norm is zero")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    q[0] = np.cos(0.5 * angle)
    q[axis + 1] = np.sin(0.5 * angle)
    return q


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return axis_quat(2, yaw)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def torso_quat(root_quat: np.ndarray, joint_pos_sdk: np.ndarray) -> np.ndarray:
    q = normalize_quat(root_quat)
    q = quat_mul(q, axis_quat(2, joint_pos_sdk[12]))
    q = quat_mul(q, axis_quat(0, joint_pos_sdk[13]))
    q = quat_mul(q, axis_quat(1, joint_pos_sdk[14]))
    return normalize_quat(q)


def anchor_orientation_obs(
    real_torso: np.ndarray, ref_torso: np.ndarray, init_quat: np.ndarray
) -> np.ndarray:
    aligned_ref = quat_mul(init_quat, ref_torso)
    relative = quat_mul(quat_conjugate(aligned_ref), real_torso)
    rot = quat_to_matrix(relative).T
    return np.array(
        [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]],
        dtype=np.float32,
    )


def load_vector(config: dict, key: str, size: int) -> np.ndarray:
    value = np.asarray(config[key], dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"{key} must contain {size} values, got {value.shape}")
    return value


def build_observation(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    motion: np.ndarray,
    motion_vel: np.ndarray,
    frame: int,
    joint_ids_map: np.ndarray,
    default_joint_pos: np.ndarray,
    last_action: np.ndarray,
    init_quat: np.ndarray,
) -> np.ndarray:
    q_sdk = data.qpos[7:].copy()
    dq_sdk = data.qvel[6:].copy()
    q_bfs = q_sdk[joint_ids_map]
    dq_bfs = dq_sdk[joint_ids_map]

    ref_q_sdk = motion[frame, 7:]
    ref_dq_sdk = motion_vel[frame]
    ref_q_bfs = ref_q_sdk[joint_ids_map]
    ref_dq_bfs = ref_dq_sdk[joint_ids_map]

    real_torso = torso_quat(data.qpos[3:7], q_sdk)
    ref_root_wxyz = motion[frame, [6, 3, 4, 5]]
    ref_torso = torso_quat(ref_root_wxyz, ref_q_sdk)
    orientation = anchor_orientation_obs(real_torso, ref_torso, init_quat)

    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-angular-velocity"
    )
    if gyro_id < 0:
        raise ValueError("Model does not contain imu-torso-angular-velocity sensor")
    gyro_start = model.sensor_adr[gyro_id]
    base_ang_vel = data.sensordata[gyro_start : gyro_start + 3].copy()

    observation = np.concatenate(
        [
            ref_q_bfs,
            ref_dq_bfs,
            orientation,
            base_ang_vel,
            q_bfs - default_joint_pos,
            dq_bfs,
            last_action,
        ]
    ).astype(np.float32)
    if observation.shape != (154,) or not np.isfinite(observation).all():
        raise RuntimeError(f"Invalid policy observation: {observation.shape}")
    return observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Input5 mimic policy in MuJoCo.")
    parser.add_argument("--model", type=Path, required=True, help="G1 29-DoF MJCF XML")
    parser.add_argument("--policy", type=Path, required=True, help="Exported ONNX policy")
    parser.add_argument("--config", type=Path, required=True, help="Isaac deploy.yaml")
    parser.add_argument("--motion", type=Path, required=True, help="50 Hz reference CSV")
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    joint_ids_map = np.asarray(config["joint_ids_map"], dtype=np.int64)
    if sorted(joint_ids_map.tolist()) != list(range(29)):
        raise ValueError("joint_ids_map must be a permutation of 0..28")

    policy_dt = float(config["step_dt"])
    substeps = round(policy_dt / args.physics_dt)
    if substeps < 1 or not np.isclose(substeps * args.physics_dt, policy_dt):
        raise ValueError("step_dt must be an integer multiple of physics-dt")

    stiffness_bfs = load_vector(config, "stiffness", 29)
    damping_bfs = load_vector(config, "damping", 29)
    default_bfs = load_vector(config, "default_joint_pos", 29)
    action_cfg = config["actions"]["JointPositionAction"]
    action_scale = np.asarray(action_cfg["scale"], dtype=np.float64)
    action_offset = np.asarray(action_cfg["offset"], dtype=np.float64)

    stiffness_sdk = np.empty(29, dtype=np.float64)
    damping_sdk = np.empty(29, dtype=np.float64)
    stiffness_sdk[joint_ids_map] = stiffness_bfs
    damping_sdk[joint_ids_map] = damping_bfs

    motion = np.loadtxt(args.motion, delimiter=",", dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"Motion CSV must have shape (frames, 36), got {motion.shape}")
    motion_vel = np.gradient(motion[:, 7:], policy_dt, axis=0)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(f"Expected G1 dimensions (36, 35, 29), got {(model.nq, model.nv, model.nu)}")
    model.opt.timestep = args.physics_dt
    data = mujoco.MjData(model)

    data.qpos[:3] = motion[0, :3]
    data.qpos[3:7] = motion[0, [6, 3, 4, 5]]
    data.qpos[7:] = motion[0, 7:]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    ref_torso_0 = torso_quat(motion[0, [6, 3, 4, 5]], motion[0, 7:])
    real_torso_0 = torso_quat(data.qpos[3:7], data.qpos[7:])
    init_quat = quat_mul(yaw_quat(real_torso_0), quat_conjugate(yaw_quat(ref_torso_0)))

    session = ort.InferenceSession(str(args.policy.resolve()), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    last_action = np.zeros(29, dtype=np.float32)
    target_sdk = motion[0, 7:].copy()
    force_limits = model.jnt_actfrcrange[1:30].copy()

    def step_policy(frame: int) -> None:
        nonlocal last_action, target_sdk
        observation = build_observation(
            data,
            model,
            motion,
            motion_vel,
            frame,
            joint_ids_map,
            default_bfs,
            last_action,
            init_quat,
        )
        output = session.run([output_name], {input_meta.name: observation[None]})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid policy output: {action.shape}")
        target_bfs = action.astype(np.float64) * action_scale + action_offset
        target_sdk = np.empty(29, dtype=np.float64)
        target_sdk[joint_ids_map] = target_bfs
        last_action = action

    def physics_step() -> None:
        q_sdk = data.qpos[7:]
        dq_sdk = data.qvel[6:]
        torque = stiffness_sdk * (target_sdk - q_sdk) - damping_sdk * dq_sdk
        torque = np.clip(torque, force_limits[:, 0], force_limits[:, 1])
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)

    print(
        f"Loaded {len(motion)} frames ({(len(motion) - 1) * policy_dt:.2f}s), "
        f"policy={input_meta.shape} -> {session.get_outputs()[0].shape}"
    )

    if args.headless:
        for frame in range(len(motion)):
            step_policy(frame)
            for _ in range(substeps):
                physics_step()
        print(f"Completed. Final pelvis height: {data.qpos[2]:.3f} m")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0
        next_tick = time.perf_counter()
        for frame in range(len(motion)):
            if not viewer.is_running():
                break
            step_policy(frame)
            for _ in range(substeps):
                physics_step()
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            if not args.no_realtime:
                next_tick += policy_dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))


if __name__ == "__main__":
    main()
