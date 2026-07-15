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

NOSE = 0
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LHEEL, RHEEL = 29, 30
LFOOT, RFOOT = 31, 32

LIMBS = [
    (LS, LE, (55, 85, 230), 0.035), (LE, LW, (70, 105, 245), 0.030),
    (RS, RE, (235, 150, 45), 0.035), (RE, RW, (245, 170, 65), 0.030),
    (LH, LK, (55, 170, 235), 0.043), (LK, LA, (80, 190, 245), 0.036),
    (RH, RK, (70, 210, 150), 0.043), (RK, RA, (95, 225, 170), 0.036),
]
JOINTS = [LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA]


def load_segments(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["fps"], [(s["start_frame"], s["end_frame"]) for s in payload["segments"]]


def landmark_array(landmarks):
    pts = np.zeros((33, 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        # Scene axes: x right, y depth, z up. Preserve video left/right in x.
        pts[i] = np.array([lm.x - 0.5, -lm.z, 1.0 - lm.y], dtype=np.float64)
    return pts


def normalize_pose(raw):
    shoulder_mid = (raw[LS] + raw[RS]) * 0.5
    hip_mid = (raw[LH] + raw[RH]) * 0.5
    torso_len = np.linalg.norm((shoulder_mid - hip_mid)[[0, 2]])
    scale = 0.62 / max(torso_len, 1e-6)
    pts = (raw - hip_mid) * scale

    # Keep original front-facing horizontal orientation, but center the actor.
    hip_after = (pts[LH] + pts[RH]) * 0.5
    pts[:, 0] -= hip_after[0]
    pts[:, 1] -= hip_after[1] * 0.25

    # Floor lock using ankle/heel/toe foot plates.
    foot_ids = [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT]
    floor_z = min(float(pts[idx, 2]) for idx in foot_ids)
    pts[:, 2] -= floor_z

    # Build neck/head from torso axis, not from nose alone.
    shoulder_mid = (pts[LS] + pts[RS]) * 0.5
    hip_mid = (pts[LH] + pts[RH]) * 0.5
    up = shoulder_mid - hip_mid
    up[1] *= 0.25
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = up / up_norm
    neck = shoulder_mid + up * 0.08
    head = shoulder_mid + up * 0.24
    pts[NOSE] = head
    return pts, neck, head


def smooth_items(items, radius=3):
    if radius <= 0 or len(items) < 3:
        return items
    poses = [item[0] for item in items]
    necks = [item[1] for item in items]
    heads = [item[2] for item in items]
    smoothed = []
    for idx in range(len(items)):
        start = max(0, idx - radius)
        end = min(len(items), idx + radius + 1)
        pts = np.mean(poses[start:end], axis=0)
        neck = np.mean(necks[start:end], axis=0)
        head = np.mean(heads[start:end], axis=0)
        floor_z = min(float(pts[i, 2]) for i in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT])
        pts[:, 2] -= floor_z
        neck[2] -= floor_z
        head[2] -= floor_z
        smoothed.append((pts, neck, head))
    return smoothed


def rotation_matrix(yaw_deg=-24, pitch_deg=-8):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return rx @ rz


def project(point, width, height, rot, distance=3.4, focal=660.0):
    q = rot @ point
    depth = distance + q[1]
    x = width * 0.5 + focal * q[0] / max(depth, 0.25)
    y = height * 0.80 - focal * q[2] / max(depth, 0.25)
    return np.array([x, y, depth], dtype=np.float64)


def shade(color, depth, min_depth=2.2, max_depth=4.8):
    t = (depth - min_depth) / max(max_depth - min_depth, 1e-6)
    factor = 1.12 - 0.32 * np.clip(t, 0, 1)
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def draw_capsule(canvas, p1, p2, radius, color, rot):
    h, w = canvas.shape[:2]
    a = project(p1, w, h, rot)
    b = project(p2, w, h, rot)
    avg_depth = (a[2] + b[2]) * 0.5
    px_radius = max(4, int(round(radius * 660 / max(avg_depth, 0.25))))
    col = shade(color, avg_depth)
    cv2.line(canvas, tuple(a[:2].astype(int)), tuple(b[:2].astype(int)), col, px_radius * 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(a[:2].astype(int)), px_radius, col, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(b[:2].astype(int)), px_radius, col, -1, cv2.LINE_AA)
    cv2.line(canvas, tuple(a[:2].astype(int)), tuple(b[:2].astype(int)), (40, 40, 40), 2, cv2.LINE_AA)


def draw_sphere(canvas, center, radius, color, rot):
    h, w = canvas.shape[:2]
    p = project(center, w, h, rot)
    px_radius = max(5, int(round(radius * 660 / max(p[2], 0.25))))
    col = shade(color, p[2])
    cv2.circle(canvas, tuple(p[:2].astype(int)), px_radius, col, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(p[:2].astype(int)), px_radius, (35, 35, 35), 2, cv2.LINE_AA)
    highlight = tuple(int(c + (255 - c) * 0.35) for c in col)
    cv2.circle(canvas, tuple((p[:2] + np.array([-px_radius * 0.3, -px_radius * 0.35])).astype(int)), max(2, px_radius // 4), highlight, -1, cv2.LINE_AA)


def draw_box(canvas, center, axes, half_sizes, color, rot):
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(center + axes[:, 0] * half_sizes[0] * sx + axes[:, 1] * half_sizes[1] * sy + axes[:, 2] * half_sizes[2] * sz)
    h, w = canvas.shape[:2]
    proj = [project(c, w, h, rot) for c in corners]
    faces = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
    ]
    face_items = []
    for face in faces:
        depth = float(np.mean([proj[i][2] for i in face]))
        poly = np.array([proj[i][:2] for i in face], dtype=np.int32)
        face_items.append((depth, poly))
    for depth, poly in sorted(face_items, reverse=True):
        cv2.fillConvexPoly(canvas, poly, shade(color, depth))
        cv2.polylines(canvas, [poly], True, (45, 45, 45), 1, cv2.LINE_AA)


def unit(v, fallback):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return fallback.copy()
    return v / n


def draw_torso(canvas, pts, rot):
    shoulder_mid = (pts[LS] + pts[RS]) * 0.5
    hip_mid = (pts[LH] + pts[RH]) * 0.5
    y_axis = unit(pts[LS] - pts[RS], np.array([1.0, 0.0, 0.0]))
    z_axis = unit(shoulder_mid - hip_mid, np.array([0.0, 0.0, 1.0]))
    x_axis = unit(np.cross(y_axis, z_axis), np.array([0.0, 1.0, 0.0]))
    y_axis = unit(np.cross(z_axis, x_axis), y_axis)
    axes = np.stack([x_axis, y_axis, z_axis], axis=1)
    center = (shoulder_mid + hip_mid) * 0.5
    half_height = np.linalg.norm(shoulder_mid - hip_mid) * 0.55
    half_width = max(np.linalg.norm(pts[LS] - pts[RS]) * 0.52, 0.13)
    draw_box(canvas, center, axes, np.array([0.055, half_width, half_height]), (90, 115, 220), rot)


def foot_box_axes(ankle, heel, toe):
    forward = unit(toe - heel, np.array([0.12, 0.0, 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    side = unit(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    up = unit(np.cross(forward, side), up)
    axes = np.stack([forward, side, up], axis=1)
    center = (ankle * 0.35 + heel * 0.25 + toe * 0.40)
    center[2] = max(center[2], 0.025)
    return center, axes


def draw_floor(canvas, rot):
    h, w = canvas.shape[:2]
    for x in np.linspace(-1.4, 1.4, 8):
        p1 = project(np.array([x, -0.8, 0]), w, h, rot)
        p2 = project(np.array([x, 0.8, 0]), w, h, rot)
        cv2.line(canvas, tuple(p1[:2].astype(int)), tuple(p2[:2].astype(int)), (225, 225, 225), 1, cv2.LINE_AA)
    for y in np.linspace(-0.8, 0.8, 6):
        p1 = project(np.array([-1.4, y, 0]), w, h, rot)
        p2 = project(np.array([1.4, y, 0]), w, h, rot)
        cv2.line(canvas, tuple(p1[:2].astype(int)), tuple(p2[:2].astype(int)), (225, 225, 225), 1, cv2.LINE_AA)


def draw_model(canvas, pts, neck, head, label):
    rot = rotation_matrix(-24, -8)
    draw_floor(canvas, rot)

    # Depth sort approximate: far legs/arms first, torso/head last.
    limb_items = []
    for a, b, col, r in LIMBS:
        depth = (project(pts[a], canvas.shape[1], canvas.shape[0], rot)[2] + project(pts[b], canvas.shape[1], canvas.shape[0], rot)[2]) * 0.5
        limb_items.append((depth, a, b, col, r))
    for _, a, b, col, r in sorted(limb_items, reverse=True):
        draw_capsule(canvas, pts[a], pts[b], r, col, rot)

    # Feet as boxes, not dots.
    for ankle, heel, toe, col in [(LA, LHEEL, LFOOT, (40, 85, 210)), (RA, RHEEL, RFOOT, (220, 140, 40))]:
        center, axes = foot_box_axes(pts[ankle], pts[heel], pts[toe])
        draw_box(canvas, center, axes, np.array([0.13, 0.045, 0.022]), col, rot)

    draw_torso(canvas, pts, rot)
    draw_capsule(canvas, neck, head, 0.028, (80, 80, 80), rot)
    draw_sphere(canvas, head, 0.105, (226, 190, 158), rot)

    for idx in JOINTS:
        draw_sphere(canvas, pts[idx], 0.035 if idx not in [LW, RW, LA, RA] else 0.032, (245, 245, 245), rot)

    cv2.rectangle(canvas, (18, 18), (740, 64), (255, 255, 255), -1)
    cv2.putText(canvas, label, (30, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)


def collect(video, model, segments, fps):
    keep = set()
    segment_of = {}
    for seg_idx, (start, end) in enumerate(segments, start=1):
        for frame in range(start, end):
            keep.add(frame)
            segment_of[frame] = seg_idx
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    items = []
    meta = []
    with PoseLandmarker.create_from_options(options) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in keep:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image(image_format=ImageFormat.SRGB, data=rgb)
                ts = int(round(frame_idx * 1000.0 / fps))
                result = pose.detect_for_video(image, ts)
                if result.pose_landmarks:
                    items.append(normalize_pose(landmark_array(result.pose_landmarks[0])))
                    meta.append((segment_of[frame_idx], frame_idx, frame_idx / fps))
            frame_idx += 1
    cap.release()
    return items, meta


def smooth(items, radius=3):
    if radius <= 0 or len(items) < 3:
        return items
    poses = [i[0] for i in items]
    necks = [i[1] for i in items]
    heads = [i[2] for i in items]
    out = []
    for i in range(len(items)):
        start = max(0, i - radius)
        end = min(len(items), i + radius + 1)
        pts = np.mean(poses[start:end], axis=0)
        neck = np.mean(necks[start:end], axis=0)
        head = np.mean(heads[start:end], axis=0)
        floor_z = min(float(pts[idx, 2]) for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT])
        pts[:, 2] -= floor_z
        neck[2] -= floor_z
        head[2] -= floor_z
        out.append((pts, neck, head))
    return out


def main():
    parser = argparse.ArgumentParser(description="Render selected dance segments as a floor-locked 3D capsule humanoid.")
    parser.add_argument("--video", default="assets/newone.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--segments", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output", default="outputs/newone_3d_capsule_model.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    source_fps, segments = load_segments(args.segments)
    items, meta = collect(args.video, args.model, segments, source_fps)
    items = smooth(items, radius=3)
    if args.stride > 1:
        items = items[:: args.stride]
        meta = meta[:: args.stride]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(out_path)

    rendered = 0
    last_segment = None
    floor_zs = []
    head_neck = []
    for (pts, neck, head), (segment, frame_idx, seconds) in zip(items, meta):
        if last_segment is not None and segment != last_segment:
            for _ in range(12):
                blank = np.full((args.height, args.width, 3), 248, np.uint8)
                cv2.putText(blank, "next selected segment", (315, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (45, 45, 45), 3, cv2.LINE_AA)
                writer.write(blank)
                rendered += 1
        last_segment = segment
        canvas = np.full((args.height, args.width, 3), 248, np.uint8)
        label = f"3D capsule retarget | segment {segment} | frame {frame_idx} | {seconds:.2f}s"
        draw_model(canvas, pts, neck, head, label)
        writer.write(canvas)
        rendered += 1
        floor_zs.append(min(float(pts[idx, 2]) for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT]))
        head_neck.append(float(np.linalg.norm(head - neck)))
        if rendered % 80 == 0:
            print(f"Rendered {rendered} frames")
    writer.release()

    stats = {
        "rendered_frames": rendered,
        "pose_frames": len(items),
        "min_floor_z": min(floor_zs) if floor_zs else None,
        "max_floor_z": max(floor_zs) if floor_zs else None,
        "min_head_neck_distance": min(head_neck) if head_neck else None,
        "max_head_neck_distance": max(head_neck) if head_neck else None,
    }
    Path("outputs/newone_3d_capsule_model_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
