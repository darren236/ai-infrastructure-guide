# Guide 02 helper scripts

[Repository overview](../../README.md#guides-and-agendas) · [Guide 02 learning path](../../docs/guides/02-nemotron-streaming-singapore-english.md#hands-on-agenda) · [Full-data extension](full-finetune/README.md)

## Two separate workflows

- **Historical 300/50 POC:** The scripts in this directory support the completed nine-stage Guide 02 journey. Keep this path frozen when reproducing its evidence.
- **Full-data extension:** The scripts under [`full-finetune/`](full-finetune/README.md) define a separate, prepared experiment with a larger-node resource gate. They do not replace or retroactively change the historical POC.

Repository presence is not deployment proof. The reported runs used `/home/ubuntu/work/nemotron-poc`; a checkout or deliberate staging step is required on a compute host.

## Historical POC script map

| Stage | Script | Purpose |
| --- | --- | --- |
| 4 / 5 | [`check_split_overlap.py`](check_split_overlap.py) | Report record/speaker counts and cross-split speaker or utterance-ID overlap. |
| 4 / 5 | [`inspect_transcript_tags.py`](inspect_transcript_tags.py) | Inventory transcript annotation tags in one or more JSONL manifests. |
| 4 | [`check_poc_eligibility.py`](check_poc_eligibility.py) | Measure the record and duration impact of excluding whole `<unk>` utterances. |
| 5 | [`create_poc_subsets.py`](create_poc_subsets.py) | Create deterministic speaker-aware 300/50 source-format subsets. |
| 5 | [`normalize_transcripts.py`](normalize_transcripts.py) | Remove only approved noise tags into a new manifest; refuse unexpected tags or overwrites. |
| 5 | [`convert_to_nemo_manifest.py`](convert_to_nemo_manifest.py) | Validate audio paths and create the five-field Nemotron manifests. |
| 6 | [`baseline_smoke.py`](baseline_smoke.py) | Restore pretrained Nemotron on GPU and transcribe the first prepared development record. |
| 6 / 8 / 9 | [`baseline_eval.py`](baseline_eval.py) | Evaluate pretrained or local `.nemo` models with the fixed offline corpus-WER policy. |
| 7 / 8 | [`run_nsc_train.sh`](run_nsc_train.sh) | Launch NVIDIA's official fine-tuning entry point for the historical smoke or pilot configuration. |

Use the linked [Guide 02 stages](../../docs/guides/02-nemotron-streaming-singapore-english.md#hands-on-agenda) for the question each script answers, the recorded result, and the evidence boundary.

## Runtime boundaries

- The six preparation helpers use Python's standard library and accept manifest paths at runtime.
- Preparation helpers do not decode or modify audio, run ASR, or overwrite existing outputs.
- `baseline_smoke.py` and `baseline_eval.py` require PyTorch and NeMo in the pinned Speech container.
- `baseline_eval.py` writes predictions and metrics to a caller-selected output directory; keep transcript-bearing outputs outside Git.
- `run_nsc_train.sh` expects the reported host layout under `~/work`, `~/data/nsc`, and `~/hf-cache`, with source data and cached model inputs mounted read-only.
- A commit does not copy scripts to the Brev workspace or prove that a remote file matches the repository version.

## Historical training modes

```bash
bash run_nsc_train.sh smoke
bash run_nsc_train.sh pilot
```

- `smoke` completed two optimizer steps and proved the official data/training/validation/export path.
- `pilot` completed 100 steps and produced the export frozen as POC v1.
- The launcher's configurable `full <steps>` mode is retained only as historical compatibility. It was not the full-data experiment and should not be used for new larger runs.

For new training beyond the POC, use the [gated full-data workflow](full-finetune/README.md). It adds prerequisite checks, duration/OOM profiling, resource telemetry, resumable segments, native streaming evaluation, development-only model selection, and candidate freezing.

## What not to repeat

- Do not create another one-record smoke script; `baseline_smoke.py` already covers that gate.
- Do not duplicate the evaluator for each checkpoint; pass `--model-path` and a distinct `--run-name` to the shared evaluator.
- Do not use final five-field NeMo manifests for speaker-overlap checks because they intentionally omit `id` and `speaker`; use the earlier source-format manifests.
- Do not treat copying a script, changing ownership, or counting lines as a learning milestone unless it resolves a concrete failure.
- Do not run the legacy `full` launcher mode as a substitute for the prepared resource-gated workflow.
