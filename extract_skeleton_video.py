import argparse
import time
from pathlib import Path

import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)


def _score(value: float | None) -> float:
    return 1.0 if value is None else value


def draw_pose_landmarks(frame, landmarks, min_landmark_score: float) -> None:
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        visible = _score(landmark.visibility) >= min_landmark_score
        present = _score(landmark.presence) >= min_landmark_score
        in_frame = 0 <= x < width and 0 <= y < height
        points.append((x, y, visible and present and in_frame))

    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[connection.start]
        end = points[connection.end]
        if start[2] and end[2]:
            cv2.line(frame, start[:2], end[:2], (255, 80, 0), 3, cv2.LINE_AA)

    for x, y, usable in points:
        if usable:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose skeleton video.")
    parser.add_argument(
        "--input",
        default="assets/newone.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/newone_skeleton.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="MediaPipe Pose Landmarker .task model path.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum pose tracking confidence.",
    )
    parser.add_argument(
        "--min-landmark-score",
        type=float,
        default=0.35,
        help="Minimum visibility/presence score used while drawing landmarks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing MediaPipe model file: {model_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    processed = 0
    detected = 0
    started = time.time()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    with PoseLandmarker.create_from_options(options) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and processed >= args.max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image(image_format=ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(processed * 1000.0 / fps))
            result = pose.detect_for_video(image, timestamp_ms)

            annotated = frame.copy()
            if result.pose_landmarks and result.pose_landmarks[0]:
                detected += 1
                draw_pose_landmarks(
                    annotated,
                    result.pose_landmarks[0],
                    args.min_landmark_score,
                )

            writer.write(annotated)
            processed += 1

            if processed % 30 == 0:
                total = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
                print(f"Processed {processed}/{total} frames, detected pose in {detected} frames")

    cap.release()
    writer.release()

    elapsed = time.time() - started
    rate = detected / processed if processed else 0.0
    print(f"Saved: {output_path}")
    print(f"Frames: {processed}, pose-detected frames: {detected}, detection rate: {rate:.1%}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
