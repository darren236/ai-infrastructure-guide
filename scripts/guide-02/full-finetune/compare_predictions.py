"""Compare ASR prediction files with paired bootstrap confidence intervals."""

import argparse
import json
import re
from pathlib import Path


LANGUAGE_TAG = re.compile(r"<[a-z]{2}(?:-[a-z]{2})?>", re.IGNORECASE)


def normalize_text(text):
    text = LANGUAGE_TAG.sub(" ", text)
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for row, ref_word in enumerate(reference, start=1):
        current = [row]
        for column, pred_word in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_word != pred_word),
                )
            )
        previous = current
    return previous[-1]


def read_run(path):
    references = []
    predictions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            reference = record.get("reference_raw", record.get("text"))
            prediction = record.get("prediction_raw", record.get("pred_text"))
            if reference is None or prediction is None:
                raise ValueError(f"{path}:{line_number}: unsupported prediction schema")
            references.append(normalize_text(reference))
            predictions.append(normalize_text(prediction))
    if not references:
        raise ValueError(f"No predictions in {path}")
    return references, predictions


def bootstrap_delta(candidate_errors, other_errors, word_counts, samples, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    deltas = []
    count = len(word_counts)
    for start in range(0, samples, 500):
        batch = min(500, samples - start)
        indices = rng.integers(0, count, size=(batch, count))
        denominators = word_counts[indices].sum(axis=1)
        candidate_wer = candidate_errors[indices].sum(axis=1) / denominators
        other_wer = other_errors[indices].sum(axis=1) / denominators
        deltas.append((candidate_wer - other_wer) * 100)
    values = np.concatenate(deltas)
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=PREDICTIONS.jsonl"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--must-beat", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for paired bootstrap scoring") from error
    run_paths = {}
    for value in args.run:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"Invalid --run value: {value}")
        if label in run_paths:
            raise ValueError(f"Duplicate run label: {label}")
        run_paths[label] = Path(raw_path)

    if args.candidate not in run_paths:
        raise ValueError("Candidate label is not present in --run inputs")
    missing = set(args.must_beat) - set(run_paths)
    if missing:
        raise ValueError(f"Unknown --must-beat labels: {sorted(missing)}")

    shared_references = None
    word_counts = None
    errors = {}
    metrics = {}
    for label, path in run_paths.items():
        references, predictions = read_run(path)
        if shared_references is None:
            shared_references = references
            word_counts = np.array(
                [len(reference.split()) for reference in references], dtype=np.int64
            )
            if np.any(word_counts == 0):
                raise ValueError("At least one normalized reference is empty")
        elif references != shared_references:
            raise ValueError(f"Reference/order mismatch in {path}")

        run_errors = np.array(
            [
                edit_distance(reference.split(), prediction.split())
                for reference, prediction in zip(references, predictions)
            ],
            dtype=np.int64,
        )
        errors[label] = run_errors
        total_errors = int(run_errors.sum())
        total_words = int(word_counts.sum())
        metrics[label] = {
            "predictions": str(path),
            "utterances": len(references),
            "reference_words": total_words,
            "word_errors": total_errors,
            "wer": total_errors / total_words,
            "wer_percent": total_errors / total_words * 100,
        }

    candidate_errors = errors[args.candidate]
    comparisons = {}
    for label, other_errors in errors.items():
        if label == args.candidate:
            continue
        point_delta = (
            metrics[args.candidate]["wer_percent"] - metrics[label]["wer_percent"]
        )
        comparisons[f"{args.candidate}_minus_{label}"] = {
            "absolute_wer_delta_percentage_points": point_delta,
            "paired_bootstrap_95_percent_ci": bootstrap_delta(
                candidate_errors,
                other_errors,
                word_counts,
                args.bootstrap_samples,
                args.seed,
            ),
        }

    passed = all(
        metrics[args.candidate]["wer"] < metrics[label]["wer"]
        for label in args.must_beat
    )
    report = {
        "dataset": args.dataset,
        "condition": args.condition,
        "normalization": (
            "strip language tags, lowercase, normalize curly apostrophes, "
            "remove punctuation except apostrophes, normalize whitespace"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "runs": metrics,
        "comparisons": comparisons,
        "acceptance": {
            "candidate": args.candidate,
            "must_beat": args.must_beat,
            "passed": passed,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.dataset}-{args.condition}-comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create the WER chart") from error

    labels = list(metrics)
    values = [metrics[label]["wer_percent"] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#777777", "#4c78a8", "#76b900"][: len(labels)])
    axis.set_ylabel("WER (%)")
    axis.set_title(f"{args.dataset}: {args.condition}")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}%", ha="center", va="bottom")
    figure.tight_layout()
    chart_path = args.output_dir / f"{args.dataset}-{args.condition}-wer.png"
    figure.savefig(chart_path, dpi=160)

    print(json.dumps(report, indent=2))
    print(chart_path)
    if args.must_beat and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
