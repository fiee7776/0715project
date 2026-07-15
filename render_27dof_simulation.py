import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

JOINT_ORDER = [
    "root_yaw", "torso_roll", "torso_pitch",
    "left_shoulder_yaw", "left_shoulder_roll", "left_shoulder_pitch", "left_elbow", "left_wrist_yaw", "left_wrist_pitch",
    "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow", "right_wrist_yaw", "right_wrist_pitch",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
]


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "segment": int(row["segment"]),
                "source_frame": int(row["source_frame"]),
                "angles": [float(row[name]) for name in JOINT_ORDER],
            })
    return rows


def setup_world(urdf_path):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=80)
    p.loadURDF("plane.urdf")
    rid = p.loadURDF(urdf_path, basePosition=[0, 0, 1.04], useFixedBase=True)
    p.changeVisualShape(rid, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
    joint_map = {}
    for idx in range(p.getNumJoints(rid)):
        info = p.getJointInfo(rid, idx)
        name = info[1].decode("utf-8")
        if name in JOINT_ORDER:
            joint_map[name] = idx
            p.setJointMotorControl2(rid, idx, p.POSITION_CONTROL, targetPosition=0, force=250)
    missing = [name for name in JOINT_ORDER if name not in joint_map]
    if missing:
        raise RuntimeError(f"URDF missing joints: {missing}")
    return cid, rid, joint_map


def set_pose(rid, joint_map, angles_deg):
    for name, angle in zip(JOINT_ORDER, angles_deg):
        p.setJointMotorControl2(
            rid,
            joint_map[name],
            p.POSITION_CONTROL,
            targetPosition=float(np.deg2rad(angle)),
            targetVelocity=0.0,
            force=320,
            positionGain=0.55,
            velocityGain=0.75,
        )


def render(width, height):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.02, 0, 1.0],
        distance=2.35,
        yaw=38,
        pitch=-10,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=42, aspect=width / height, nearVal=0.05, farVal=10)
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    frame = np.asarray(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw_label(frame, row):
    label = f"Segment {row['segment']} | source frame {row['source_frame']}"
    cv2.rectangle(frame, (18, 18), (470, 60), (255, 255, 255), -1)
    cv2.putText(frame, label, (30, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Render selected 27-DOF humanoid dance segments with PyBullet.")
    parser.add_argument("--csv", default="outputs/newone_22dof_selected.csv")
    parser.add_argument("--urdf", default="assets/dance_humanoid_22dof.urdf")
    parser.add_argument("--output", default="outputs/newone_27dof_simulation_selected.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth mapped frame to make preview shorter.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--pause-frames", type=int, default=18)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if args.stride > 1:
        rows = rows[:: args.stride]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(output)

    cid, rid, joint_map = setup_world(args.urdf)
    last_segment = None
    rendered = 0
    try:
        for idx, row in enumerate(rows):
            if last_segment is not None and row["segment"] != last_segment:
                for _ in range(args.pause_frames):
                    frame = render(args.width, args.height)
                    cv2.putText(frame, "next selected segment", (310, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 3, cv2.LINE_AA)
                    writer.write(frame)
                    rendered += 1
            last_segment = row["segment"]
            set_pose(rid, joint_map, row["angles"])
            for _ in range(8):
                p.stepSimulation()
            frame = render(args.width, args.height)
            draw_label(frame, row)
            writer.write(frame)
            rendered += 1
            if (idx + 1) % 60 == 0:
                print(f"Rendered {idx + 1}/{len(rows)} pose frames")
    finally:
        writer.release()
        p.disconnect(cid)

    print(f"Saved: {output}")
    print(f"Rendered video frames: {rendered}")


if __name__ == "__main__":
    main()


