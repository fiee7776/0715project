import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

# MediaPipe indices.
NOSE = 0
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LHEEL, RHEEL = 29, 30
LFOOT, RFOOT = 31, 32

SEGMENTS = [
    (LS, RS), (LS, LH), (RS, RH), (LH, RH),
    (LS, LE), (LE, LW), (RS, RE), (RE, RW),
    (LH, LK), (LK, LA), (RH, RK), (RK, RA),
    (LA, LFOOT), (RA, RFOOT),
    (LS, NOSE), (RS, NOSE),
]

LEFT_PARTS = {LS, LE, LW, LH, LK, LA, LHEEL, LFOOT}
RIGHT_PARTS = {RS, RE, RW, RH, RK, RA, RHEEL, RFOOT}


def load_segments(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["fps"], [(s["start_frame"], s["end_frame"]) for s in payload["segments"]]


def mp_to_scene(landmarks):
    pts = np.zeros((33, 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        # X: horizontal, Y: depth-ish, Z: up.
        pts[i] = np.array([lm.x - 0.5, -lm.z, 1.0 - lm.y], dtype=np.float64)
    return pts


def normalize_pose(points):
    hip_mid = (points[LH] + points[RH]) * 0.5
    shoulder_mid = (points[LS] + points[RS]) * 0.5
    torso_len = np.linalg.norm(shoulder_mid - hip_mid)
    scale = 0.62 / max(torso_len, 1e-6)
    pts = (points - hip_mid) * scale

    foot_indices = [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT]
    floor_z = min(float(pts[idx, 2]) for idx in foot_indices)
    pts[:, 2] -= floor_z

    # Keep the dancer roughly centered but let hip/torso bounce and sway remain visible.
    hip_after = (pts[LH] + pts[RH]) * 0.5
    pts[:, 0] -= hip_after[0] * 0.35
    pts[:, 1] -= hip_after[1] * 0.35
    return pts


def smooth_poses(poses, radius=3):
    if radius <= 0 or len(poses) < 3:
        return poses
    out = []
    for i in range(len(poses)):
        start = max(0, i - radius)
        end = min(len(poses), i + radius + 1)
        out.append(np.mean(poses[start:end], axis=0))
    # Re-apply floor lock after smoothing so feet do not sink.
    locked = []
    for pts in out:
        pts = pts.copy()
        floor_z = min(float(pts[idx, 2]) for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT])
        pts[:, 2] -= floor_z
        locked.append(pts)
    return locked


def project(point, width, height, yaw, pitch, distance):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = point
    # Rotate around Z then X.
    x1 = cy * x - sy * y
    y1 = sy * x + cy * y
    z1 = z
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    depth = distance + y2
    focal = 620.0
    u = width * 0.5 + focal * x1 / max(depth, 0.2)
    v = height * 0.78 - focal * z2 / max(depth, 0.2)
    return int(round(u)), int(round(v)), depth


def color_for_joint(idx):
    if idx in LEFT_PARTS:
        return (235, 74, 64)
    if idx in RIGHT_PARTS:
        return (45, 150, 245)
    return (30, 30, 30)


def draw_stick_robot(canvas, pts, label):
    height, width = canvas.shape[:2]
    yaw = math.radians(-22)
    pitch = math.radians(-7)
    projected = [project(p, width, height, yaw, pitch, 3.4) for p in pts]

    # Floor grid.
    floor_y = int(height * 0.78)
    cv2.line(canvas, (0, floor_y), (width, floor_y), (170, 170, 170), 2, cv2.LINE_AA)
    for x in range(-6, 7):
        p1 = project(np.array([x * 0.25, -0.75, 0]), width, height, yaw, pitch, 3.4)
        p2 = project(np.array([x * 0.25, 0.75, 0]), width, height, yaw, pitch, 3.4)
        cv2.line(canvas, p1[:2], p2[:2], (220, 220, 220), 1, cv2.LINE_AA)
    for y in range(-4, 5):
        p1 = project(np.array([-1.6, y * 0.25, 0]), width, height, yaw, pitch, 3.4)
        p2 = project(np.array([1.6, y * 0.25, 0]), width, height, yaw, pitch, 3.4)
        cv2.line(canvas, p1[:2], p2[:2], (225, 225, 225), 1, cv2.LINE_AA)

    # Draw limb shadows first.
    for a, b in SEGMENTS:
        pa = projected[a]
        pb = projected[b]
        shadow_a = (pa[0], floor_y + 4)
        shadow_b = (pb[0], floor_y + 4)
        cv2.line(canvas, shadow_a, shadow_b, (210, 210, 210), 4, cv2.LINE_AA)

    # Torso fill as a quadrilateral.
    torso_poly = np.array([projected[LS][:2], projected[RS][:2], projected[RH][:2], projected[LH][:2]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, torso_poly, (208, 219, 248))
    cv2.polylines(canvas, [torso_poly], True, (55, 75, 145), 3, cv2.LINE_AA)

    for a, b in SEGMENTS:
        pa = projected[a]
        pb = projected[b]
        color = color_for_joint(a)
        thickness = 13 if {a, b} <= {LS, RS, LH, RH} else 10
        cv2.line(canvas, pa[:2], pb[:2], color, thickness, cv2.LINE_AA)
        cv2.line(canvas, pa[:2], pb[:2], (35, 35, 35), 2, cv2.LINE_AA)

    # Head.
    nose = projected[NOSE]
    shoulder_mid = (pts[LS] + pts[RS]) * 0.5
    head_center_3d = pts[NOSE] * 0.72 + shoulder_mid * 0.28 + np.array([0, 0, 0.08])
    head = project(head_center_3d, width, height, yaw, pitch, 3.4)
    cv2.circle(canvas, head[:2], 28, (222, 190, 160), -1, cv2.LINE_AA)
    cv2.circle(canvas, head[:2], 28, (35, 35, 35), 2, cv2.LINE_AA)

    # Joint balls, bigger for feet to reveal floor contact.
    for idx, p2 in enumerate(projected):
        if idx not in [NOSE, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA, LFOOT, RFOOT]:
            continue
        radius = 8 if idx not in [LA, RA, LFOOT, RFOOT] else 11
        cv2.circle(canvas, p2[:2], radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2[:2], radius, color_for_joint(idx), 3, cv2.LINE_AA)

    # Ground contact markers.
    for idx in [LA, RA, LFOOT, RFOOT]:
        p2 = projected[idx]
        cv2.circle(canvas, (p2[0], floor_y), 7, (40, 150, 70), -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (18, 18), (650, 64), (255, 255, 255), -1)
    cv2.putText(canvas, label, (30, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)


def collect_poses(video_path, model_path, segments, fps):
    keep = set()
    segment_of = {}
    for seg_idx, (start, end) in enumerate(segments, start=1):
        for frame in range(start, end):
            keep.add(frame)
            segment_of[frame] = seg_idx

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    poses = []
    meta = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in keep:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image(image_format=ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_idx * 1000.0 / fps))
                result = landmarker.detect_for_video(image, timestamp_ms)
                if result.pose_landmarks:
                    pts = normalize_pose(mp_to_scene(result.pose_landmarks[0]))
                    poses.append(pts)
                    meta.append((segment_of[frame_idx], frame_idx, frame_idx / fps))
            frame_idx += 1
    cap.release()
    return poses, meta


def main():
    parser = argparse.ArgumentParser(description="Render a readable floor-locked humanoid retargeting preview.")
    parser.add_argument("--video", default="assets/newone.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--segments", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output", default="outputs/newone_floor_locked_retarget_preview.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    source_fps, segments = load_segments(args.segments)
    poses, meta = collect_poses(args.video, args.model, segments, source_fps)
    poses = smooth_poses(poses, radius=3)

    if args.stride > 1:
        poses = poses[:: args.stride]
        meta = meta[:: args.stride]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(out)

    last_segment = None
    rendered = 0
    floor_mins = []
    hip_heights = []
    for pts, (segment, frame_idx, seconds) in zip(poses, meta):
        if last_segment is not None and segment != last_segment:
            for _ in range(12):
                canvas = np.full((args.height, args.width, 3), 248, np.uint8)
                cv2.putText(canvas, "next selected segment", (315, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (45, 45, 45), 3, cv2.LINE_AA)
                writer.write(canvas)
                rendered += 1
        last_segment = segment
        canvas = np.full((args.height, args.width, 3), 248, np.uint8)
        label = f"Floor-locked retarget | segment {segment} | source frame {frame_idx} | {seconds:.2f}s"
        draw_stick_robot(canvas, pts, label)
        writer.write(canvas)
        rendered += 1
        floor_mins.append(float(min(pts[idx, 2] for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT])))
        hip_heights.append(float(((pts[LH] + pts[RH]) * 0.5)[2]))
        if rendered % 80 == 0:
            print(f"Rendered {rendered} frames")
    writer.release()

    stats = {
        "rendered_frames": rendered,
        "pose_frames": len(poses),
        "min_floor_z": min(floor_mins) if floor_mins else None,
        "max_floor_z": max(floor_mins) if floor_mins else None,
        "min_hip_height": min(hip_heights) if hip_heights else None,
        "max_hip_height": max(hip_heights) if hip_heights else None,
    }
    Path("outputs/newone_floor_locked_retarget_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
