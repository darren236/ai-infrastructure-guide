"""
Guide 02 — Nemotron offline validation evaluation.

Evaluate either the original pretrained model or a local .nemo export.
Preserve the transcription and scoring policy of the recorded baseline.

Outputs:
    <run_name>_predictions.jsonl
    <run_name>_metrics.json

Original datasets, manifests, and checkpoints are only read.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)
from nemo.collections.asr.metrics.wer import word_error_rate


MODEL_NAME = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Keep this expression identical to the recorded baseline.
LANGUAGE_TAG = re.compile(
    r"<[a-z]{2}(?:-[a-z]{2})?>",
    flags=re.IGNORECASE,
)


def normalize_text(text):
    """Apply the original baseline policy; no number/abbreviation rules."""
    text = LANGUAGE_TAG.sub(" ", text)
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = " ".join(text.split())
    return text


def load_manifest(path):
    """Load all nonempty JSONL records without modifying the manifest."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # 1. Select the manifest, model source, and output label.
    parser = argparse.ArgumentParser(
        description="Evaluate Nemotron with the fixed baseline scoring policy."
    )
    parser.add_argument(
        "manifest",
        help="Path to validation JSONL manifest",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local .nemo export; omit to load the original pretrained model",
    )
    parser.add_argument(
        "--run-name",
        default="baseline",
        help="Output filename label; default: baseline",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for evaluation results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Read the same fixed validation manifest.
    samples = load_manifest(args.manifest)
    if not samples:
        raise ValueError("Validation manifest is empty.")

    print("\n=== DATASET ===")
    print(f"Manifest : {args.manifest}")
    print(f"Records  : {len(samples)}")

    languages = {sample["target_lang"] for sample in samples}
    if languages != {"en-US"}:
        raise ValueError(
            f"Expected en-US recordings, found: {languages}"
        )

    audio_paths = [sample["audio_filepath"] for sample in samples]
    references = [sample["text"] for sample in samples]

    # 3. Restore the selected model. This does not resume training.
    if args.model_path:
        print(f"\nRestoring local model: {args.model_path}")
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=args.model_path
        )
    else:
        print("\nLoading pretrained Nemotron...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL_NAME
        )

    model.cuda()
    model.eval()

    model_class = type(model).__name__
    model_device = str(next(model.parameters()).device)
    print(f"Model class : {model_class}")
    print(f"Model device: {model_device}")

    # 4. Preserve the exact successful baseline transcription arguments.
    transcribe_cfg = RNNTPromptTranscribeConfig(
        use_lhotse=False,
        batch_size=1,
        num_workers=0,
        target_lang="en-US",
        verbose=False,
    )

    # 5. Perform offline whole-file transcription.
    print(f"\nTranscribing {len(audio_paths)} recordings...")
    with torch.inference_mode():
        results = model.transcribe(
            audio=audio_paths,
            override_config=transcribe_cfg,
        )

    if len(results) != len(samples):
        raise RuntimeError(
            f"Expected {len(samples)} predictions, "
            f"received {len(results)}."
        )

    predictions = [
        result.text if hasattr(result, "text") else str(result)
        for result in results
    ]

    # 6. Normalize references and predictions with the original policy.
    normalized_refs = [normalize_text(text) for text in references]
    normalized_preds = [normalize_text(text) for text in predictions]

    for i, reference in enumerate(normalized_refs):
        if not reference:
            raise ValueError(
                f"Empty normalized reference at record {i + 1}"
            )

    # 7. Calculate corpus WER, not mean per-recording WER.
    wer = word_error_rate(
        hypotheses=normalized_preds,
        references=normalized_refs,
    )
    wer_percent = float(wer) * 100
    total_words = sum(len(text.split()) for text in normalized_refs)

    # 8. Preserve both raw and normalized prediction/reference pairs.
    predictions_path = (
        output_dir / f"{args.run_name}_predictions.jsonl"
    )
    with open(predictions_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            record = {
                "audio_filepath": sample["audio_filepath"],
                "duration": sample["duration"],
                "target_lang": sample["target_lang"],
                "reference_raw": references[i],
                "prediction_raw": predictions[i],
                "reference_normalized": normalized_refs[i],
                "prediction_normalized": normalized_preds[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 9. Identify the evaluated artifact and scoring settings.
    metrics = {
        "model": MODEL_NAME,
        "base_model": MODEL_NAME,
        "model_path": args.model_path,
        "model_source": (
            "local_nemo" if args.model_path else "pretrained"
        ),
        "model_class": model_class,
        "model_device": model_device,
        "run_name": args.run_name,
        "manifest": args.manifest,
        "evaluation_mode": "offline_transcribe",
        "num_utterances": len(samples),
        "total_reference_words": total_words,
        "target_lang": "en-US",
        "batch_size": 1,
        "num_workers": 0,
        "use_lhotse": False,
        "verbose": False,
        "normalization": (
            "strip language tags, lowercase, "
            "normalize curly apostrophes, "
            "remove punctuation except apostrophes, "
            "normalize whitespace; "
            "no number or abbreviation normalization"
        ),
        "wer": float(wer),
        "wer_percent": wer_percent,
    }

    metrics_path = output_dir / f"{args.run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 10. Print results for the operator to inspect.
    print("\n=== EVALUATION RESULTS ===")
    print(f"Run name       : {args.run_name}")
    print(f"Utterances     : {len(samples)}")
    print(f"Reference words: {total_words}")
    print(f"WER            : {wer_percent:.2f}%")

    print("\n=== FIRST 5 PREDICTIONS ===")
    for i in range(min(5, len(samples))):
        print(f"\n[{i + 1}]")
        print(f"REF : {normalized_refs[i]}")
        print(f"PRED: {normalized_preds[i]}")

    print("\n=== SAVED FILES ===")
    print(predictions_path)
    print(metrics_path)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
