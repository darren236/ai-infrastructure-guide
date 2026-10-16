# Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md) · [Helper scripts](../../scripts/guide-02/README.md) · [Learning Record](../../PROGRESS.md)

## Goal and scope

Use a small, reproducible proof of concept to answer one question: **does adapting `nvidia/nemotron-3.5-asr-streaming-0.6b` to Singapore English improve recognition enough to justify a larger experiment?**

The core journey uses an NVIDIA L4, a pinned NeMo Speech container, deterministic NSC Part 6 train/development inputs, the official NVIDIA training entry point, fixed offline scoring, and one frozen held-out comparison. It demonstrates a sound evaluation sequence; it does not claim production quality, statistical significance, streaming performance, or deployment readiness.

**Evidence basis:** Results are operator-reported from October 5–14, 2026. The repository maintainer did not independently run the GPU experiments or inspect the remote artifacts. Raw audio, predictions, checkpoints, model weights, caches, and logs remain outside Git.

<a id="customer-requirement-and-deployment-architecture"></a>

## Architecture and responsibilities

| Layer | Recorded choice | Why it matters |
| --- | --- | --- |
| Compute | Brev Ubuntu node with one NVIDIA L4 | Keeps the exercise bounded to available hardware. |
| Runtime | `nvcr.io/nvidia/nemo-speech:26.07.00`, NeMo `3.0.0` | Pins the demonstrated userspace stack. |
| Model | `nvidia/nemotron-3.5-asr-streaming-0.6b` | Starts from the pretrained streaming-capable checkpoint. |
| Training code | Speech `v3.0.0` `speech_to_text_finetune.py` | Uses NVIDIA's supported recipe instead of a custom training loop. |
| Data | NSC Part 6 | Separates training, development selection, and held-out testing. |
| Evaluation | Fixed corpus WER policy | Makes pretrained and adapted results comparable. |
| Persistence | Host-mounted cache and results directories | Keeps model downloads and artifacts outside disposable containers. |

Launch Docker from the Ubuntu host. Mount source data and code read-only, mount only the result destination writable, and keep container paths consistent with paths stored in the NeMo manifests. A container tag improves repeatability but is not an immutable image digest.

<a id="solutions-architect-decision-framework"></a>

## Decision gates

An SA does not need every possible check. Each gate below exists to answer a different decision:

1. **Environment gate:** Can the selected image restore the model on the available GPU?
2. **Data gate:** Are train, development, and test roles explicit and free of known speaker/utterance leakage?
3. **Baseline gate:** Is there a fixed, reproducible comparison before training?
4. **Training gate:** Does the official path complete, save, and restore an artifact?
5. **Selection gate:** Does a candidate improve fixed development WER enough to freeze it?
6. **Held-out gate:** Does the frozen candidate retain an improvement on unseen in-domain data?
7. **Scale gate:** Does the larger workflow fit the target node with measured headroom?

Do not repeat inventory commands, checksum every file, reproduce identical outputs, or add another smoke path once the relevant gate is already answered. Repeat a check only when the environment, artifact, policy, or failure symptom changes.

<a id="guide-sections"></a>
<a id="current-pipeline-status"></a>
<a id="hands-on-agenda"></a>

## Core learning path

1. [Node resources](#1-node-resources) — establish a clean starting snapshot. **Verified.**
2. [NeMo container](#2-nemo-container) — confirm the pinned runtime imports. **Verified.**
3. [Pretrained model](#3-pretrained-model) — restore Nemotron and place it on the L4. **Verified.**
4. [Data strategy and source checks](#4-data-strategy-and-source-checks) — define roles and check separation. **Verified.**
5. [Training and validation inputs](#5-training-and-validation-inputs) — build deterministic, normalized NeMo manifests. **Verified.**
6. [Pretrained inference and baseline](#6-pretrained-inference-and-baseline) — prove transcription and record fixed-development WER. **Verified.**
7. [Training smoke test and artifacts](#7-training-smoke-test-and-artifacts) — exercise the official training and export path. **Verified.**
8. [Checkpoint comparison and development](#8-checkpoint-comparison-and-development) — compare candidates and freeze POC v1. **Verified.**
9. [Held-out NSC test](#9-held-out-nsc-test) — evaluate the frozen candidate without retuning. **Verified.**

The nine-stage core is complete. The separate full-data workflow, native streaming comparison, and external benchmark are further learning with independent evidence gates.

<a id="node-preflight--complete"></a>

## 1. Node resources

**Question:** Is the node ready to load the model and begin a small experiment?

Use the minimum snapshot that rules out obvious capacity or contention problems:

```bash
df -h
free -h
nvidia-smi
```

The recorded node had 214 GB free disk, approximately 14 GiB available host RAM, no swap, and an idle NVIDIA L4 with 23,034 MiB VRAM. This was enough to begin model loading and the historical 300/50 POC; it did not prove capacity for a larger training run.

**What we learned:** A preflight is a point-in-time triage check, not workload sizing. Once the environment is stable, do not repeat it before every small command unless resource state may have changed.

<a id="nemo-container-and-model-on-gpu--complete"></a>

## 2. NeMo container

**Question:** Does the selected image provide the expected NeMo runtime and GPU access?

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo; print(nemo.__version__)"
```

**Observed:** NeMo `3.0.0` imported successfully. Earlier guides already established Docker and NVIDIA GPU passthrough, so repeating lower-layer host checks here would not add evidence unless this command failed.

## 3. Pretrained model

**Question:** Can the original model be restored on the selected GPU?

Use a persistent host cache so model downloads survive container removal:

```bash
mkdir -p ~/hf-cache

docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo.collections.asr as n; m=n.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b'); m.cuda(); print(type(m).__name__); print(next(m.parameters()).device)"
```

**Observed:** `EncDecRNNTBPEModelWithPrompt` restored and its parameters moved to `cuda:0`.

**What we learned:** Model restoration and placement prove runtime compatibility for loading. They do not prove decoding quality, training capacity, or throughput.

<a id="data-strategy-and-experiment-overview"></a>
<a id="split-construction-and-hands-on-verification"></a>
<a id="transcript-annotation-audit--complete"></a>
<a id="poc-eligibility-impact--complete"></a>

## 4. Data strategy and source checks

**Question:** Which data may influence weights, model selection, and the final claim?

| Data | Role | Model learns from it? | Decision use |
| --- | --- | --- | --- |
| `nsc_query_5h` | Training | Yes | Weight updates. |
| `nsc_dev_3h` | Development/validation | No gradients | Candidate and configuration selection. |
| `nsc_test` | Held-out in-domain test | No | One frozen comparison after development decisions. |
| `gigaspeech_test` | Optional external benchmark | No | Generalization/regression check; never routine tuning. |

Run the read-only helpers before deriving inputs:

```bash
TRAIN_MANIFEST=/path/to/nsc_query_5h/manifest.jsonl
DEV_MANIFEST=/path/to/nsc_dev_3h/manifest.jsonl

python scripts/guide-02/check_split_overlap.py "$TRAIN_MANIFEST" "$DEV_MANIFEST"
python scripts/guide-02/inspect_transcript_tags.py "$TRAIN_MANIFEST" "$DEV_MANIFEST"
python scripts/guide-02/check_poc_eligibility.py "$TRAIN_MANIFEST" "$DEV_MANIFEST"
```

The source manifests contained 2,289 train records / 111 speakers and 1,316 development records / 64 speakers, with no shared speakers or utterance IDs. The audit found `<unk>`, `<v-noise>`, and `<noise>` annotations.

The declared POC policy was:

- exclude whole `<unk>` utterances rather than leave speech without a reliable reference;
- keep utterances containing `<v-noise>` or `<noise>`, then remove only those tokens in a derived transcript;
- preserve words, fillers, case, punctuation, metadata, source manifests, and audio;
- avoid manual relabeling in this reproducible exercise.

This left eligible pools of 2,158 train records / 4.59 hours and 1,225 development records / 2.71 hours.

**What we learned:** Split roles and reference policy affect the credibility of the result more than another round of file-count checks. Once the declared invariants pass, move to the next decision gate.

<a id="deterministic-poc-subset-generation--complete"></a>
<a id="derived-poc-subset-validation--complete"></a>
<a id="derived-annotation-tags"></a>
<a id="derived-speaker-and-utterance-id-separation"></a>
<a id="transcript-annotation-normalization--complete"></a>
<a id="nemotron-compatible-manifest-conversion--complete"></a>

## 5. Training and validation inputs

**Question:** Can the experiment inputs be recreated without modifying source data?

Use the repository helpers in order. Output directories must already exist, and the writers refuse to overwrite existing destinations.

```bash
TRAIN_MANIFEST=/path/to/nsc_query_5h/manifest.jsonl
DEV_MANIFEST=/path/to/nsc_dev_3h/manifest.jsonl
TRAIN_AUDIO_ROOT=/path/to/nsc_query_5h
DEV_AUDIO_ROOT=/path/to/nsc_dev_3h
SUBSET_DIR=/path/to/poc
NORMALIZED_DIR="$SUBSET_DIR/normalized"
NEMO_DIR="$SUBSET_DIR/nemo"

mkdir -p "$SUBSET_DIR" "$NORMALIZED_DIR" "$NEMO_DIR"

python scripts/guide-02/create_poc_subsets.py \
  "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$SUBSET_DIR" \
  --train-count 300 --dev-count 50 --seed 42

python scripts/guide-02/normalize_transcripts.py \
  "$SUBSET_DIR/train_300.jsonl" "$NORMALIZED_DIR/train_300.jsonl"
python scripts/guide-02/normalize_transcripts.py \
  "$SUBSET_DIR/dev_50.jsonl" "$NORMALIZED_DIR/dev_50.jsonl"

python scripts/guide-02/convert_to_nemo_manifest.py \
  "$NORMALIZED_DIR/train_300.jsonl" "$TRAIN_AUDIO_ROOT" "$NEMO_DIR/train_300.jsonl"
python scripts/guide-02/convert_to_nemo_manifest.py \
  "$NORMALIZED_DIR/dev_50.jsonl" "$DEV_AUDIO_ROOT" "$NEMO_DIR/dev_50.jsonl"
```

**Recorded outputs:**

- training: 300 recordings, all 111 eligible train speakers represented, approximately 0.66 hours;
- development: 50 recordings from 50 speakers, approximately 0.11 hours;
- zero shared speakers or utterance IDs;
- 61 train and four development transcripts changed by declared noise-token removal;
- NeMo manifests contain `audio_filepath`, `duration`, `text`, `lang`, and `target_lang`, with `en-US` conditioning.

Keep source, selected, normalized, and NeMo-format manifests as separate stages. Rechecking every invariant after each format conversion is unnecessary; check only the invariants that the transformation can change.

<a id="single-utterance-pretrained-gpu-inference--complete"></a>
<a id="pretrained-50-utterance-validation-baseline--complete"></a>
<a id="reported-evaluation-implementation-and-scoring-policy"></a>

## 6. Pretrained inference and baseline

**Question:** Does the model decode real input, and what is the fixed comparison before training?

Run the helpers inside the pinned NeMo container with the cache, data, scripts, and result directory mounted appropriately:

```bash
python /work/baseline_smoke.py /data/nsc/poc/nemo/dev_50.jsonl

python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --run-name pretrained \
  --output-dir /results
```

The one-record smoke test produced a nonempty transcript on `cuda:0`. The fixed 50-record evaluation contained 793 normalized reference words and scored **10.844892812105927% offline corpus WER** for the pretrained model.

The evaluator keeps the successful prompt-aware configuration (`en-US`, batch size one, `use_lhotse=False`) and applies one fixed normalization policy: remove language tags, lowercase, normalize the curly apostrophe, replace other punctuation with spaces, and collapse whitespace. It preserves raw and normalized prediction/reference pairs outside Git.

**What we learned:** One transcription proves the application path. The fixed baseline answers the quality question. A second ad hoc audio smoke test would be redundant unless it targets a different failure mode.

<a id="official-nemo-fine-tuning--two-step-gpu-smoke-test"></a>
<a id="troubleshooting-history-failures-and-fixes-in-order"></a>

## 7. Training smoke test and artifacts

**Question:** Does the official training route complete and produce a restorable artifact?

The historical launcher uses Speech `v3.0.0` and NVIDIA's `speech_to_text_finetune.py`:

```bash
bash scripts/guide-02/run_nsc_train.sh smoke
```

The recorded configuration used AdamW at `1e-5`, `bf16-mixed`, batch size one, `max_duration=39.99`, one L4, two optimizer steps, and limited validation. Run `smoke-2-20261009T092152Z` produced checkpoints and a `.nemo` export on persistent host storage.

**What we learned:** A two-step smoke test validates configuration, data loading, forward/backward execution, validation entry, saving, and export. It does not establish convergence, full-data consumption, resource headroom, or model improvement. An SA should stop the smoke run once those mechanics are proven.

<a id="next-restore-and-evaluate-the-exported-checkpoint--planned"></a>
<a id="export-evaluation-and-baseline-reproduction--complete"></a>
<a id="offline-comparison-and-interpretation"></a>
<a id="pre-training-readiness-review--complete"></a>
<a id="controlled-100-step-pilot--complete"></a>
<a id="standardized-pilot-export-evaluation"></a>
<a id="recording-level-error-comparison--complete"></a>
<a id="next-investigate-training-behavior-and-validation-logs--planned"></a>
<a id="lessons-learned-for-customer-decisions"></a>
<a id="evidence-identities-captured"></a>

## 8. Checkpoint comparison and development

**Question:** Does an adapted artifact improve on the fixed development set enough to freeze it?

Evaluate an export with the same manifest and policy by adding `--model-path`:

```bash
python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --model-path /results/model.nemo \
  --run-name candidate \
  --output-dir /results/evaluation
```

| Model | Training steps | Errors / 793 words | Offline WER |
| --- | ---: | ---: | ---: |
| Pretrained | 0 | 86 | 10.844892812105927% |
| Smoke export | 2 | 87 | 10.970996216897856% |
| Pilot export | 100 | 83 | 10.466582597730138% |

The smoke export was slightly worse, as expected for a mechanics test. The controlled 100-step pilot completed on October 10, 2026 and reduced development errors from 86 to 83. Recording-level counts were nine improved, seven worsened, and 34 equal-error recordings; equal counts do not imply identical transcripts.

The pilot export was frozen as **POC v1** before held-out inference. Its SHA-256 is `e81ae310dc076e2a74ea4e31b7f45e696f0d062379ff7d3a9ba4636b251bd863`.

Known limitations remain: step-50 validation history, exact data exposure/repetition, full loss/resource history, and best/last/export weight provenance were not resolved. Reconstructing every missing training detail would not change the bounded POC decision; those gaps should instead become requirements for the next experiment.

<a id="held-out-results"></a>

## 9. Held-out NSC test

**Question:** Does frozen POC v1 retain an improvement on unseen in-domain data?

The official source contained 3,684 records / FLAC files and 70 speakers, with no speaker or utterance-ID overlap against the train or development sources. Before inference, the declared reference policy removed `<unk>`, `<v-noise>`, and `<noise>`, collapsed whitespace, and excluded six references left empty. A targeted decoder scan excluded eight FLAC files that failed in the evaluation container. Both models used the same final 3,670-record manifest.

| Model | Errors / 64,490 words | Offline WER |
| --- | ---: | ---: |
| Pretrained | 8,104 | 12.566289347185611% |
| POC v1 | 7,903 | 12.254613118312916% |

POC v1 made **201 fewer errors**, an improvement of **0.311676 percentage points absolute** and **2.480257% relative**. The final manifest SHA-256 is `99a6057842ff72ca990c0639a53da44a5be1a6ed1e3224c9710bcfeb13e71809`.

**What we learned:** The direction of the development result held on the declared in-domain test, but the improvement is modest. No significance interval, business-critical error review, streaming comparison, or production performance result supports a broader claim. Do not tune POC v1 from this test result.

<a id="10-external-benchmark"></a>

## Further learning

### Full-data fine-tuning

The separate [full-data runbook](../../scripts/guide-02/full-finetune/README.md) uses 2,152 train recordings / 4.50 hours and 1,203 development recordings / 2.65 hours. The prepared manifests have SHA-256 values `e80e8795a2c009e12496bd3fafeb3b4ba0569f738d6a7f4f323eaed64dba69fe` and `a5a84863e904b093c814ee9742ef8de30e5032f7cb9d24c665965744bcd8680d`; exclusions are recorded in [reviewed evidence](../evidence/guide-02-full-finetune-exclusions.csv).

Two attempts in Docker containers capped at 10 GiB reached five steps and exited `137`; the kernel identified memory-cgroup OOM kills. Do not keep guessing smaller batches on the same 12 GiB node. The next useful gate is a 32 GiB-or-larger node, NeMo duration/OOM profiling, and an independent 50-step resource pilot. Full training proceeds only if that gate passes.

### Streaming and external evaluation

The prepared workflow makes native 320 ms cache-aware streaming the primary development condition and retains offline scoring for continuity. A new candidate must beat both pretrained and POC v1 on streaming development WER before freezing. The already-observed NSC test becomes a reused benchmark for that new experiment, not a pristine test.

`gigaspeech_test` remains an optional external/OOD benchmark. Use it only after freezing and do not tune from it. Latency, throughput, concurrency, memory headroom under load, and deployment cost also remain unmeasured.

<a id="completion-boundaries-and-next-step"></a>

## What we learned and where the evidence stops

- A pinned container and official recipe reduced environment ambiguity without requiring a host ML stack.
- Deterministic data lineage and split discipline made the evaluation interpretable.
- Smoke testing proved mechanics; fixed development WER supported selection; frozen held-out WER supported a bounded claim.
- The 100-step candidate improved modestly on development and held-out NSC data.
- The historical run is useful evidence, not a template for production sizing or statistical claims.
- The larger experiment needs an explicit resource gate instead of more trial-and-error on an undersized node.

<a id="troubleshooting-notes"></a>
<a id="missing-runtime-helper"></a>
<a id="appendix-optional-single-file-inference-smoke-test--not-yet-executed"></a>

## Avoid redundant work

- **Do not repeat lower-layer GPU checks after the application is already running** unless the environment or failure changes.
- **Do not maintain a second one-WAV smoke path.** The verified manifest-based smoke helper already exercises real input and the prompt-aware configuration.
- **Do not treat file ownership, line counts, and copy commands as learning stages.** They are operational fixes or deployment details, not architecture evidence.
- **Do not rerun the baseline merely to seek byte-identical predictions.** Rerun it when the model, manifest, decoding, normalization, container, or comparison policy changes.
- **Do not investigate every training-log omission before a bounded POC decision.** Carry unresolved provenance and resource requirements into the next experiment instead.
- **Do not use the held-out test to select checkpoints or tune scoring.** Once observed, it cannot regain pristine-test status for later candidates.

The [helper-script index](../../scripts/guide-02/README.md) documents the supported repository entry points. The [Learning Record](../../PROGRESS.md) keeps the concise chronology; prior command-by-command detail remains available in Git history.
