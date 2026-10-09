"""Run the reported one-record pretrained Nemotron GPU inference workflow."""

import argparse
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)


MODEL_NAME = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe the first recording in a validation JSONL manifest."
    )
    parser.add_argument("validation_manifest", type=Path)
    args = parser.parse_args()

    # Read only the first recording. This is not the 50-utterance WER baseline.
    # Paths must be visible inside the container, e.g. under /data/nsc.
    with args.validation_manifest.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            if line.strip():
                recording = json.loads(line)
                break
        else:
            raise ValueError("Validation manifest contains no recordings")

    audio_path = recording["audio_filepath"]
    reference = recording["text"]
    target_lang = recording["target_lang"]
    # Do not silently substitute a language when conditioning metadata is absent.
    if not isinstance(target_lang, str) or not target_lang.strip():
        raise ValueError("The first recording needs a nonempty target_lang")

    # Restore the original checkpoint, then enable GPU inference/evaluation.
    # The host-mounted Hugging Face cache persists after the container exits.
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    model.cuda()
    model.eval()
    print(f"Model class : {type(model).__name__}")
    print(f"Model device: {next(model.parameters()).device}")

    # Preserve the full configuration that resolved the observed prompt error.
    # Disabling Lhotse is a successful workaround, not proof of its root cause.
    transcribe_cfg = RNNTPromptTranscribeConfig(
        use_lhotse=False,
        batch_size=1,
        num_workers=0,
        target_lang=target_lang,
        verbose=False,
    )

    print("\nRunning inference...")
    # Evaluation mode and inference_mode serve different purposes:
    # eval selects evaluation behavior; inference_mode disables gradient tracking.
    with torch.inference_mode():
        results = model.transcribe(
            audio=[audio_path],
            override_config=transcribe_cfg,
        )

    if not results:
        raise RuntimeError("Transcription returned no result")
    # Preserve the raw text, including any emitted language marker.
    # Accommodate a text string or a NeMo hypothesis without changing decoding.
    prediction = getattr(results[0], "text", results[0])
    if not isinstance(prediction, str) or not prediction.strip():
        raise RuntimeError("Transcription returned no nonempty text")

    print("\n=== RESULTS ===")
    print(f"Reference : {reference}")
    print(f"Prediction: {prediction}")
    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
