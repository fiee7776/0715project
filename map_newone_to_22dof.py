import argparse
import csv
import json
import math
from pathlib import Path

import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

JOINT_ORDER = [
    "root_yaw", "torso_roll", "torso_pitch",
    "left_shoulder_yaw", "left_shoulder_roll", "left_shoulder_pitch", "left_elbow", "left_wrist_yaw", "left_wrist_pitch",
    "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow", "right_wrist_yaw", "right_wrist_pitch",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
]

LIMITS_DEG = {
    "root_yaw": (-45, 45), "torso_roll": (-35, 35), "torso_pitch": (-35, 35),
    "left_shoulder_yaw": (-110, 110), "left_shoulder_roll": (-135, 135), "left_shoulder_pitch": (-135, 135), "left_elbow": (0, 155), "left_wrist_yaw": (-60, 60), "left_wrist_pitch": (-55, 55),
    "right_shoulder_yaw": (-110, 110), "right_shoulder_roll": (-135, 135), "right_shoulder_pitch": (-135, 135), "right_elbow": (0, 155), "right_wrist_yaw": (-60, 60), "right_wrist_pitch": (-55, 55),
    "left_hip_yaw": (-55, 55), "left_hip_roll": (-60, 60), "left_hip_pitch": (-85, 85), "left_knee": (0, 150), "left_ankle_pitch": (-35, 35), "left_ankle_roll": (-30, 30),
    "right_hip_yaw": (-55, 55), "right_hip_roll": (-60, 60), "right_hip_pitch": (-85, 85), "right_knee": (0, 150), "right_ankle_pitch": (-35, 35), "right_ankle_roll": (-30, 30),
}

# MediaPipe landmark indices.
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LI, RI = 19, 20
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LF, RF = 31, 32


def clamp(value, low, high):
    return max(low, min(high, value))


def moving_average(rows, radius=4):
    if radius <= 0 or len(rows) <= 2:
        return rows
    out = []
    for idx in range(len(rows)):
        start = max(0, idx - radius)
        end = min(len(rows), idx + radius + 1)
        count = end - start
        out.append([sum(rows[j][k] for j in range(start, end)) / count for k in range(len(rows[idx]))])
    return out


def lm_to_robot(landmarks):
    # Robot frame: X forward/depth, Y left, Z up.
    pts = []
    for lm in landmarks:
        pts.append((-lm.z, -lm.x, -lm.y))
    return pts


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(max(dot(a, a), 1e-12))


def angle_between(a, b):
    c = clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0)
    return math.degrees(math.acos(c))


def signed_angle_2d(y, x):
    return math.degrees(math.atan2(y, x))


def vector_angles(vec, side):
    # side is +1 for left, -1 for right. Return yaw/roll/pitch-ish values that match the URDF axes.
    x, y, z = vec
    side_y = y * side
    yaw = signed_angle_2d(x, side_y) * 0.85
    roll = signed_angle_2d(z, abs(side_y) + 1e-6)
    pitch = signed_angle_2d(-x, z + 1e-6) * 0.55
    return yaw, roll, pitch


def leg_angles(thigh, side):
    x, y, z = thigh
    side_y = y * side
    yaw = signed_angle_2d(x, max(abs(side_y), 1e-6)) * 0.65
    roll = signed_angle_2d(side_y, -z + 1e-6) * 0.9
    pitch = signed_angle_2d(x, -z + 1e-6) * 0.95
    return yaw, roll, pitch


def wrist_angles(forearm, hand, side):
    fx, fy, fz = forearm
    hx, hy, hz = hand
    rel = sub(hand, scale(forearm, dot(hand, forearm) / max(dot(forearm, forearm), 1e-6)))
    yaw = signed_angle_2d(rel[0], (rel[1] * side) + 1e-6) * 0.7
    pitch = signed_angle_2d(rel[2], abs(rel[1]) + 1e-6) * 0.7
    return yaw, pitch


def ankle_angles(shin, foot, side):
    pitch = signed_angle_2d(foot[0], abs(foot[2]) + 1e-6) * 0.5
    roll = signed_angle_2d(foot[1] * side, abs(foot[2]) + 1e-6) * 0.5
    return pitch, roll


def map_frame(landmarks):
    p = lm_to_robot(landmarks)
    shoulder_mid = scale(add(p[LS], p[RS]), 0.5)
    hip_mid = scale(add(p[LH], p[RH]), 0.5)
    torso = sub(shoulder_mid, hip_mid)
    shoulder_line = sub(p[LS], p[RS])
    hip_line = sub(p[LH], p[RH])

    root_yaw = (signed_angle_2d(shoulder_line[0], shoulder_line[1] + 1e-6) - signed_angle_2d(hip_line[0], hip_line[1] + 1e-6)) * 0.4
    torso_roll = signed_angle_2d(torso[1], torso[2] + 1e-6) * 0.9
    torso_pitch = signed_angle_2d(torso[0], torso[2] + 1e-6) * 0.9

    l_upper = sub(p[LE], p[LS])
    l_fore = sub(p[LW], p[LE])
    l_hand = sub(p[LI], p[LW])
    r_upper = sub(p[RE], p[RS])
    r_fore = sub(p[RW], p[RE])
    r_hand = sub(p[RI], p[RW])

    lsy, lsr, lsp = vector_angles(l_upper, side=1)
    rsy, rsr, rsp = vector_angles(r_upper, side=-1)
    lelb = 180.0 - angle_between(scale(l_upper, -1), l_fore)
    relb = 180.0 - angle_between(scale(r_upper, -1), r_fore)
    lwy, lwp = wrist_angles(l_fore, l_hand, side=1)
    rwy, rwp = wrist_angles(r_fore, r_hand, side=-1)

    l_thigh = sub(p[LK], p[LH])
    l_shin = sub(p[LA], p[LK])
    l_foot = sub(p[LF], p[LA])
    r_thigh = sub(p[RK], p[RH])
    r_shin = sub(p[RA], p[RK])
    r_foot = sub(p[RF], p[RA])

    lhy, lhr, lhp = leg_angles(l_thigh, side=1)
    rhy, rhr, rhp = leg_angles(r_thigh, side=-1)
    lknee = 180.0 - angle_between(scale(l_thigh, -1), l_shin)
    rknee = 180.0 - angle_between(scale(r_thigh, -1), r_shin)
    lap, lar = ankle_angles(l_shin, l_foot, side=1)
    rap, rar = ankle_angles(r_shin, r_foot, side=-1)

    values = {
        "root_yaw": root_yaw, "torso_roll": torso_roll, "torso_pitch": torso_pitch,
        "left_shoulder_yaw": lsy, "left_shoulder_roll": lsr, "left_shoulder_pitch": lsp, "left_elbow": lelb, "left_wrist_yaw": lwy, "left_wrist_pitch": lwp,
        "right_shoulder_yaw": rsy, "right_shoulder_roll": rsr, "right_shoulder_pitch": rsp, "right_elbow": relb, "right_wrist_yaw": rwy, "right_wrist_pitch": rwp,
        "left_hip_yaw": lhy, "left_hip_roll": lhr, "left_hip_pitch": lhp, "left_knee": lknee, "left_ankle_pitch": lap, "left_ankle_roll": lar,
        "right_hip_yaw": rhy, "right_hip_roll": rhr, "right_hip_pitch": rhp, "right_knee": rknee, "right_ankle_pitch": rap, "right_ankle_roll": rar,
    }
    return [clamp(values[name], *LIMITS_DEG[name]) for name in JOINT_ORDER]


def parse_segments(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["fps"], [(s["start_frame"], s["end_frame"]) for s in payload["segments"]]


def main():
    parser = argparse.ArgumentParser(description="Extract selected pose segments and map them to 22-DOF robot joint CSV.")
    parser.add_argument("--input", default="assets/newone.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--segments", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output", default="outputs/newone_22dof_selected.csv")
    parser.add_argument("--smooth-radius", type=int, default=5)
    args = parser.parse_args()

    source_fps, segments = parse_segments(args.segments)
    keep = set()
    segment_by_frame = {}
    for seg_idx, (start, end) in enumerate(segments, start=1):
        for frame in range(start, end):
            keep.add(frame)
            segment_by_frame[frame] = seg_idx

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(args.input)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    meta = []
    with PoseLandmarker.create_from_options(options) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_idx * 1000.0 / source_fps))
            if frame_idx in keep:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image(image_format=ImageFormat.SRGB, data=rgb)
                result = pose.detect_for_video(image, timestamp_ms)
                if result.pose_landmarks:
                    rows.append(map_frame(result.pose_landmarks[0]))
                    meta.append((segment_by_frame[frame_idx], frame_idx, frame_idx / source_fps))
            else:
                # Keep timestamps monotonic in video mode, but skip heavy inference outside selected windows.
                pass
            frame_idx += 1
    cap.release()

    rows = moving_average(rows, radius=args.smooth_radius)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "source_frame", "source_seconds", *JOINT_ORDER])
        for (segment, source_frame, seconds), values in zip(meta, rows):
            writer.writerow([segment, source_frame, seconds, *values])

    print(f"Saved {len(rows)} mapped frames to {out}")
    print(f"Joint columns: {len(JOINT_ORDER)}")


if __name__ == "__main__":
    main()
