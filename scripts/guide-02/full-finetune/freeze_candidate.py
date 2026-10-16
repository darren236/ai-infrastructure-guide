"""Freeze a development-selected model before any reused-test evaluation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("development_report", type=Path)
    parser.add_argument("candidate_model", type=Path)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.development_report.read_text(encoding="utf-8"))
    if report.get("dataset") != "development":
        raise ValueError("Selection must use a development report")
    if report.get("condition") != "streaming_320ms":
        raise ValueError("Primary selection condition must be streaming_320ms")
    acceptance = report.get("acceptance", {})
    if acceptance.get("candidate") != args.candidate_label:
        raise ValueError("Candidate label does not match the development report")
    if not acceptance.get("passed"):
        raise ValueError("Candidate did not pass the development acceptance gate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_model = args.output_dir / "nemotron-singlish-selected.nemo"
    if frozen_model.exists():
        raise FileExistsError(f"Refusing to replace frozen model: {frozen_model}")
    shutil.copy2(args.candidate_model, frozen_model)
    frozen_model.chmod(0o444)

    selection = {
        "selected_from": str(args.development_report),
        "selection_dataset": "development",
        "selection_condition": "streaming_320ms",
        "candidate_label": args.candidate_label,
        "source_model": str(args.candidate_model),
        "frozen_model": str(frozen_model),
        "sha256": sha256(frozen_model),
        "test_results_used_for_selection": False,
    }
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
