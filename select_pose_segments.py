import argparse
import csv
import json
from pathlib import Path

import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions


CORE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def score_landmarks(landmarks):
    if not landmarks:
        return 0.0
    scores = []
    for idx in CORE_INDICES:
        lm = landmarks[idx]
        visible = 1.0 if lm.visibility is None else lm.visibility
        present = 1.0 if lm.presence is None else lm.presence
        in_frame = 1.0 if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0 else 0.0
        scores.append(max(0.0, min(visible, present)) * in_frame)
    return sum(scores) / len(scores)


def choose_segments(scores, fps, segment_seconds, min_gap_seconds, count):
    window = max(1, int(round(segment_seconds * fps)))
    gap = max(1, int(round(min_gap_seconds * fps)))
    candidates = []
    if len(scores) < window:
        return []
    prefix = [0.0]
    for value in scores:
        prefix.append(prefix[-1] + value)
    for start in range(0, len(scores) - window + 1):
        avg = (prefix[start + window] - prefix[start]) / window
        candidates.append((avg, start, start + window))
    candidates.sort(reverse=True)

    selected = []
    for avg, start, end in candidates:
        if all(end <= s - gap or start >= e + gap for _, s, e in selected):
            selected.append((avg, start, end))
            if len(selected) >= count:
                break
    return sorted(selected, key=lambda item: item[1])


def main():
    parser = argparse.ArgumentParser(description="Score pose extraction quality and choose stable dance segments.")
    parser.add_argument("--input", default="assets/newone.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--output-json", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output-csv", default="outputs/newone_pose_scores.csv")
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--min-gap-seconds", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    scores = []
    with PoseLandmarker.create_from_options(options) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image(image_format=ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            result = pose.detect_for_video(image, timestamp_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
            scores.append(score_landmarks(landmarks))
            frame_idx += 1
            if frame_idx % 120 == 0:
                print(f"Scored {frame_idx}/{frame_count} frames")
    cap.release()

    segments = choose_segments(scores, fps, args.segment_seconds, args.min_gap_seconds, args.count)
    payload = {
        "video": str(input_path),
        "fps": fps,
        "frame_count": len(scores),
        "segment_seconds": args.segment_seconds,
        "segments": [
            {
                "rank": idx + 1,
                "start_frame": start,
                "end_frame": end,
                "start_seconds": start / fps,
                "end_seconds": end / fps,
                "average_score": avg,
            }
            for idx, (avg, start, end) in enumerate(segments)
        ],
    }

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "score"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, idx / fps, score])

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
