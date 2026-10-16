"""Export scalar histories and compact training/validation plots from TensorBoard."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_tag(tags, candidates):
    for candidate in candidates:
        if candidate in tags:
            return candidate
    for tag in tags:
        if any(tag.endswith(candidate) for candidate in candidates):
            return tag
    return None


def main():
    args = parse_args()
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as error:
        raise RuntimeError("tensorboard is required to read event files") from error

    event_files = sorted(args.log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files under {args.log_dir}")

    points = defaultdict(dict)
    all_tags = set()
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            all_tags.add(tag)
            for event in accumulator.Scalars(tag):
                points[tag][event.step] = event.value

    selected = {
        "train_loss": find_tag(all_tags, ["train_loss", "loss"]),
        "val_wer": find_tag(all_tags, ["val_wer"]),
        "learning_rate": find_tag(all_tags, ["learning_rate", "lr"]),
        "step_seconds": find_tag(
            all_tags, ["train_step_timing in s", "train_step_timing"]
        ),
    }
    selected = {name: tag for name, tag in selected.items() if tag is not None}
    if "train_loss" not in selected or "val_wer" not in selected:
        raise ValueError(
            "Required train_loss/val_wer tags were not found; available tags: "
            + ", ".join(sorted(all_tags))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training-scalars.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "tensorboard_tag", "step", "value"])
        for metric, tag in selected.items():
            for step, value in sorted(points[tag].items()):
                writer.writerow([metric, tag, step, value])

    metadata = {
        "event_files": [str(path) for path in event_files],
        "selected_tags": selected,
        "available_tags": sorted(all_tags),
    }
    (args.output_dir / "training-scalar-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create training plots") from error

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for metric, axis, color in (
        ("train_loss", axes[0], "#1f77b4"),
        ("val_wer", axes[1], "#d62728"),
    ):
        tag = selected[metric]
        series = sorted(points[tag].items())
        axis.plot([step for step, _ in series], [value for _, value in series], color=color)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("Optimizer step")
    figure.suptitle("Nemotron Singlish fine-tuning")
    figure.tight_layout()
    plot_path = args.output_dir / "training-validation-curves.png"
    figure.savefig(plot_path, dpi=160)

    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
