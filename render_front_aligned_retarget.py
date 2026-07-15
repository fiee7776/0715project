import argparse
import json
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

BODY_LINES = [
    (LS, RS), (LS, LH), (RS, RH), (LH, RH),
    (LS, LE), (LE, LW), (RS, RE), (RE, RW),
    (LH, LK), (LK, LA), (RH, RK), (RK, RA),
]
LEFT = {LS, LE, LW, LH, LK, LA, LHEEL, LFOOT}
RIGHT = {RS, RE, RW, RH, RK, RA, RHEEL, RFOOT}


def load_segments(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["fps"], [(s["start_frame"], s["end_frame"]) for s in payload["segments"]]


def landmark_array(landmarks):
    pts = np.zeros((33, 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        # Keep the same front-facing screen orientation as the source video.
        pts[i] = [lm.x, lm.y, lm.z]
    return pts


def normalize_front_pose(raw):
    pts = raw.copy()
    shoulder_mid = (pts[LS] + pts[RS]) * 0.5
    hip_mid = (pts[LH] + pts[RH]) * 0.5
    torso = shoulder_mid - hip_mid
    torso_len = np.linalg.norm(torso[:2])
    scale = 0.56 / max(torso_len, 1e-6)

    # Center on hips horizontally, but keep vertical dance motion relative to the feet.
    out = np.zeros_like(pts)
    out[:, 0] = (pts[:, 0] - hip_mid[0]) * scale
    out[:, 1] = (pts[:, 1] - hip_mid[1]) * scale
    out[:, 2] = pts[:, 2] * scale

    # Build a stable head connected to the torso instead of trusting the nose directly.
    shoulder_mid2 = (out[LS] + out[RS]) * 0.5
    hip_mid2 = (out[LH] + out[RH]) * 0.5
    up = shoulder_mid2[:2] - hip_mid2[:2]
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, -1.0])
    else:
        up = up / up_norm
    neck = shoulder_mid2.copy()
    neck[:2] = shoulder_mid2[:2] + up * 0.08
    head_center = shoulder_mid2.copy()
    head_center[:2] = shoulder_mid2[:2] + up * 0.22
    out[NOSE, :2] = head_center[:2]
    out[NOSE, 2] = shoulder_mid2[2]

    # Use ankle/heel/toe as a foot plate. Lock the lowest visible foot-plate y to ground.
    foot_ids = [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT]
    lowest_y = max(float(out[idx, 1]) for idx in foot_ids)
    out[:, 1] -= lowest_y

    # Keep the dancer inside the canvas. Positive y is downward; ground is y=0.
    out[:, 1] *= -1.0
    return out, neck, head_center


def smooth_items(items, radius=3):
    if radius <= 0 or len(items) < 3:
        return items
    poses = [item[0] for item in items]
    necks = [item[1] for item in items]
    heads = [item[2] for item in items]
    smoothed = []
    for i in range(len(items)):
        start = max(0, i - radius)
        end = min(len(items), i + radius + 1)
        pts = np.mean(poses[start:end], axis=0)
        neck = np.mean(necks[start:end], axis=0)
        head = np.mean(heads[start:end], axis=0)
        lowest = min(float(pts[idx, 1]) for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT])
        pts[:, 1] -= lowest
        neck[1] -= lowest
        head[1] -= lowest
        smoothed.append((pts, neck, head))
    return smoothed


def color(idx):
    if idx in LEFT:
        return (58, 88, 235)
    if idx in RIGHT:
        return (235, 148, 40)
    return (45, 45, 45)


def to_px(point, width, height, scale_px=430):
    # Ground is near bottom; x/y orientation mirrors source video exactly.
    cx = width * 0.5
    ground = height * 0.86
    return int(round(cx + point[0] * scale_px)), int(round(ground - point[1] * scale_px))


def draw_foot(canvas, pts, ankle, heel, toe, side_color):
    a = np.array(to_px(pts[ankle], canvas.shape[1], canvas.shape[0]))
    h = np.array(to_px(pts[heel], canvas.shape[1], canvas.shape[0]))
    t = np.array(to_px(pts[toe], canvas.shape[1], canvas.shape[0]))
    # If heel/toe are noisy or too collapsed, synthesize a short foot along their direction.
    if np.linalg.norm(t - h) < 12:
        direction = t - a
        if np.linalg.norm(direction) < 1:
            direction = np.array([1, 0])
        direction = direction / max(np.linalg.norm(direction), 1)
        h = a - direction * 18
        t = a + direction * 30
    cv2.line(canvas, tuple(h.astype(int)), tuple(t.astype(int)), side_color, 16, cv2.LINE_AA)
    cv2.line(canvas, tuple(h.astype(int)), tuple(t.astype(int)), (35, 35, 35), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(a.astype(int)), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(a.astype(int)), 8, side_color, 3, cv2.LINE_AA)


def draw_front_robot(canvas, pts, neck, head, label):
    h, w = canvas.shape[:2]
    ground_y = int(h * 0.86)
    cv2.line(canvas, (0, ground_y), (w, ground_y), (160, 160, 160), 2, cv2.LINE_AA)
    for x in range(0, w, 80):
        cv2.line(canvas, (x, ground_y), (x + 45, h), (224, 224, 224), 1, cv2.LINE_AA)

    # Torso panel.
    torso_poly = np.array([
        to_px(pts[LS], w, h), to_px(pts[RS], w, h), to_px(pts[RH], w, h), to_px(pts[LH], w, h)
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, torso_poly, (214, 224, 250))
    cv2.polylines(canvas, [torso_poly], True, (48, 66, 145), 3, cv2.LINE_AA)

    for a, b in BODY_LINES:
        pa = to_px(pts[a], w, h)
        pb = to_px(pts[b], w, h)
        cv2.line(canvas, pa, pb, color(a), 11, cv2.LINE_AA)
        cv2.line(canvas, pa, pb, (35, 35, 35), 2, cv2.LINE_AA)

    # Neck and head are attached to the shoulders/torso axis.
    shoulder_mid = (pts[LS] + pts[RS]) * 0.5
    cv2.line(canvas, to_px(shoulder_mid, w, h), to_px(neck, w, h), (45, 45, 45), 8, cv2.LINE_AA)
    head_px = to_px(head, w, h)
    cv2.circle(canvas, head_px, 30, (226, 190, 158), -1, cv2.LINE_AA)
    cv2.circle(canvas, head_px, 30, (35, 35, 35), 2, cv2.LINE_AA)

    # Foot plates after legs, so foot orientation is visible.
    draw_foot(canvas, pts, LA, LHEEL, LFOOT, color(LA))
    draw_foot(canvas, pts, RA, RHEEL, RFOOT, color(RA))

    for idx in [LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA]:
        p = to_px(pts[idx], w, h)
        cv2.circle(canvas, p, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 8, color(idx), 3, cv2.LINE_AA)

    cv2.rectangle(canvas, (18, 18), (760, 64), (255, 255, 255), -1)
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
                    items.append(normalize_front_pose(landmark_array(result.pose_landmarks[0])))
                    meta.append((segment_of[frame_idx], frame_idx, frame_idx / fps))
            frame_idx += 1
    cap.release()
    return items, meta


def main():
    parser = argparse.ArgumentParser(description="Render front-view retargeting aligned with the source video orientation.")
    parser.add_argument("--video", default="assets/newone.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--segments", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output", default="outputs/newone_front_aligned_retarget.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    source_fps, segments = load_segments(args.segments)
    items, meta = collect(args.video, args.model, segments, source_fps)
    items = smooth_items(items, radius=3)
    if args.stride > 1:
        items = items[:: args.stride]
        meta = meta[:: args.stride]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(out)

    last_segment = None
    foot_y_values = []
    head_neck_dists = []
    rendered = 0
    for (pts, neck, head), (segment, frame_idx, seconds) in zip(items, meta):
        if last_segment is not None and segment != last_segment:
            for _ in range(12):
                blank = np.full((args.height, args.width, 3), 248, np.uint8)
                cv2.putText(blank, "next selected segment", (315, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (45, 45, 45), 3, cv2.LINE_AA)
                writer.write(blank)
                rendered += 1
        last_segment = segment
        canvas = np.full((args.height, args.width, 3), 248, np.uint8)
        label = f"front-aligned retarget | segment {segment} | frame {frame_idx} | {seconds:.2f}s"
        draw_front_robot(canvas, pts, neck, head, label)
        writer.write(canvas)
        rendered += 1
        foot_y_values.append(min(float(pts[idx, 1]) for idx in [LA, RA, LHEEL, RHEEL, LFOOT, RFOOT]))
        head_neck_dists.append(float(np.linalg.norm((head - neck)[:2])))
    writer.release()

    stats = {
        "rendered_frames": rendered,
        "pose_frames": len(items),
        "min_grounded_foot_y": min(foot_y_values) if foot_y_values else None,
        "max_grounded_foot_y": max(foot_y_values) if foot_y_values else None,
        "min_head_neck_distance": min(head_neck_dists) if head_neck_dists else None,
        "max_head_neck_distance": max(head_neck_dists) if head_neck_dists else None,
    }
    Path("outputs/newone_front_aligned_retarget_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
