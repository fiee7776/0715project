from __future__ import annotations

import argparse
from statistics import fmean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_body_rot",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_joint_vel",
    "Episode_Reward/undesired_contacts",
    "Episode_Termination/time_out",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tail", type=int, default=50)
    args = parser.parse_args()

    accumulator = EventAccumulator(args.run_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))

    for tag in DEFAULT_TAGS:
        if tag not in available:
            print(f"{tag}: missing")
            continue

        events = accumulator.Scalars(tag)
        recent = events[-args.tail :]
        values = [event.value for event in recent]
        last = events[-1]
        print(
            f"{tag}: step={last.step} last={last.value:.6f} "
            f"tail{len(values)}_mean={fmean(values):.6f} "
            f"tail_min={min(values):.6f} tail_max={max(values):.6f}"
        )


if __name__ == "__main__":
    main()
