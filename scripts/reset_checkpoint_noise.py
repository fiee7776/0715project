from __future__ import annotations

import argparse
import math

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint")
    parser.add_argument("output_checkpoint")
    parser.add_argument("--std", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    args = parser.parse_args()

    if args.std <= 0.0:
        raise ValueError("--std must be positive")

    checkpoint = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    if "log_std" not in model_state:
        raise KeyError("Checkpoint does not contain model_state_dict['log_std']")

    model_state["log_std"].fill_(math.log(args.std))
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        raise ValueError("Checkpoint does not contain an optimizer state")
    for param_group in optimizer_state["param_groups"]:
        param_group["lr"] = args.lr

    torch.save(checkpoint, args.output_checkpoint)

    restored_std = model_state["log_std"].exp()
    print(
        f"saved={args.output_checkpoint} iter={checkpoint.get('iter')} "
        f"std_min={restored_std.min().item():.4f} std_max={restored_std.max().item():.4f} "
        f"lr={args.lr:.2e}"
    )


if __name__ == "__main__":
    main()
