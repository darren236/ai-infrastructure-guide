"""Summarize and plot resource telemetry from a training allocation."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.telemetry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Telemetry CSV contains no samples.")

    timestamps = [
        datetime.strptime(row["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
        for row in rows
    ]
    elapsed_minutes = [
        (timestamp - timestamps[0]).total_seconds() / 60 for timestamp in timestamps
    ]
    gpu_util = [float(row["gpu_util_percent"]) for row in rows]
    gpu_used = [float(row["gpu_memory_used_mib"]) for row in rows]
    gpu_total = [float(row["gpu_memory_total_mib"]) for row in rows]
    container_percent = [float(row["container_memory_percent"]) for row in rows]
    host_available = [float(row["host_memory_available_kib"]) for row in rows]
    host_total = [float(row["host_memory_total_kib"]) for row in rows]

    minimum_gpu_headroom = min(
        (total - used) / total for used, total in zip(gpu_used, gpu_total)
    )
    minimum_host_headroom = min(
        available / total for available, total in zip(host_available, host_total)
    )

    summary = {
        "telemetry": str(args.telemetry),
        "samples": len(rows),
        "duration_minutes": elapsed_minutes[-1],
        "mean_gpu_util_percent": sum(gpu_util) / len(gpu_util),
        "peak_gpu_memory_mib": max(gpu_used),
        "gpu_memory_total_mib": max(gpu_total),
        "minimum_gpu_headroom_percent": minimum_gpu_headroom * 100,
        "peak_container_memory_percent": max(container_percent),
        "minimum_host_memory_headroom_percent": minimum_host_headroom * 100,
        "pilot_gate": {
            "required_gpu_headroom_percent": 10.0,
            "required_host_memory_headroom_percent": 20.0,
            "passed": minimum_gpu_headroom >= 0.10
            and minimum_host_headroom >= 0.20,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "resource-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to create the resource plot"
        ) from error

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(elapsed_minutes, gpu_util, color="#76b900")
    axes[0].set_ylabel("GPU util. (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)

    axes[1].plot(elapsed_minutes, gpu_used, color="#1f77b4")
    axes[1].axhline(max(gpu_total) * 0.90, color="#d62728", linestyle="--")
    axes[1].set_ylabel("VRAM (MiB)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(elapsed_minutes, container_percent, label="container RAM")
    host_used_percent = [
        (1 - available / total) * 100
        for available, total in zip(host_available, host_total)
    ]
    axes[2].plot(elapsed_minutes, host_used_percent, label="host RAM")
    axes[2].axhline(80, color="#d62728", linestyle="--")
    axes[2].set_ylabel("Memory used (%)")
    axes[2].set_xlabel("Elapsed minutes")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.suptitle("Nemotron fine-tuning resource profile")
    figure.tight_layout()
    plot_path = args.output_dir / "resource-profile.png"
    figure.savefig(plot_path, dpi=160)

    print(json.dumps(summary, indent=2))
    print(plot_path)


if __name__ == "__main__":
    main()
