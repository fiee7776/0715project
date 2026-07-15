import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side source and retarget preview.")
    parser.add_argument("--source", default="outputs/newone_skeleton.mp4")
    parser.add_argument("--retarget", default="outputs/newone_floor_locked_retarget_preview.mp4")
    parser.add_argument("--segments", default="outputs/newone_selected_segments.json")
    parser.add_argument("--output", default="outputs/newone_floor_locked_comparison.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    import json
    payload = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    source_fps = payload["fps"]
    source_frames = []
    for segment in payload["segments"]:
        source_frames.extend(range(segment["start_frame"], segment["end_frame"], 2))
    # Match retarget pauses between segments.
    expanded_source_frames = []
    current_seg = 1
    pos = 0
    for segment in payload["segments"]:
        if current_seg > 1:
            expanded_source_frames.extend([None] * 12)
        expanded_source_frames.extend(range(segment["start_frame"], segment["end_frame"], 2))
        current_seg += 1

    src = cv2.VideoCapture(args.source)
    ret = cv2.VideoCapture(args.retarget)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(args.output)

    left_w = args.width // 2
    right_w = args.width - left_w
    frame_idx = 0
    last_src = None
    while True:
        ok_ret, ret_frame = ret.read()
        if not ok_ret:
            break
        if frame_idx < len(expanded_source_frames) and expanded_source_frames[frame_idx] is not None:
            src.set(cv2.CAP_PROP_POS_FRAMES, expanded_source_frames[frame_idx])
            ok_src, src_frame = src.read()
            if ok_src:
                last_src = src_frame
        if last_src is None:
            last_src = np.full((1280, 720, 3), 245, np.uint8)

        left = cv2.resize(last_src, (left_w, args.height))
        right = cv2.resize(ret_frame, (right_w, args.height))
        canvas = np.hstack([left, right])
        cv2.line(canvas, (left_w, 0), (left_w, args.height), (30, 30, 30), 3)
        cv2.rectangle(canvas, (20, 18), (260, 58), (255, 255, 255), -1)
        cv2.putText(canvas, "source skeleton", (32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (left_w + 20, 18), (left_w + 350, 58), (255, 255, 255), -1)
        cv2.putText(canvas, "floor-locked retarget", (left_w + 32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        writer.write(canvas)
        frame_idx += 1

    src.release()
    ret.release()
    writer.release()
    print(f"Saved: {args.output}")
    print(f"Frames: {frame_idx}")


if __name__ == "__main__":
    main()
