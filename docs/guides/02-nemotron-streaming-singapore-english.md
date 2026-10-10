# Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md)

## Goal and scope

Build a small proof of concept (POC) and tutorial for fine-tuning `nvidia/nemotron-3.5-asr-streaming-0.6b` on Singapore National Speech Corpus (NSC) Part 6, using an NVIDIA L4 24 GB class Brev instance. The goal is a manageable learning exercise, not production-quality tuning. Model loading, dataset preparation, training, and evaluation results are documented only after they are performed.

### Customer requirement and deployment architecture

**Illustrative customer scenario:** a Singapore-based organization wants to determine whether adapting Nemotron to Singapore English yields enough recognition benefit to justify engineering work and GPU spending. This is a decision framework for the learning POC, not a claim of an actual customer engagement or agreed production requirements.

| Component | Recorded technology / role |
| --- | --- |
| Compute environment | NVIDIA Brev, Ubuntu, NVIDIA L4 |
| Host runtime | NVIDIA driver, Docker with NVIDIA GPU support |
| Application container | `nvcr.io/nvidia/nemo-speech:26.07.00` |
| ML frameworks | NVIDIA NeMo / PyTorch |
| Base model | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Training entry point | NVIDIA `speech_to_text_finetune.py`, host-mounted Speech `v3.0.0` source |
| Data | NSC Part 6; 300 train / 50 development recordings |
| Quality evaluation | Offline transcription and corpus-level WER |

```text
NVIDIA Brev instance
├── Ubuntu host
│   ├── NVIDIA driver + L4 GPU
│   ├── Docker / NVIDIA GPU integration
│   ├── Project workspace, NVIDIA source, model cache
│   ├── NSC data
│   └── Persistent experiment results
└── NeMo Speech container
    ├── CUDA libraries, PyTorch, NeMo
    ├── Mounted source/data/models
    └── Training → export restoration → offline evaluation
```

Launch Docker from the Ubuntu host; **all model training, restoration, and inference execute inside the pinned NeMo Speech container**. Host checks inspect files or metadata. The tag keeps the application dependency selection consistent and reduces CUDA/framework mismatch risk without installing the ML stack into the host OS. It supports portability to compatible GPU hosts; driver compatibility, GPU capacity, and mounts still need verification. A tag is less precise than an immutable digest, which has not been recorded here.

Training maps `~/work/nemo-speech-src` to `/nemo-src:ro`, `~/data/nsc` to `/data/nsc:ro`, cached weights to `/root/.cache/huggingface:ro`, and each host experiment directory to writable `/results`. Evaluation maps the project to `/work:ro` for the evaluator/export and uses separate writable results mounts. The commands below define the exact mappings; container paths embedded in manifests must match them. Writable host mounts preserve checkpoints/metrics after `--rm`; a file stored only in the disposable container would be lost.

### Solutions Architect decision framework

| Decision | Customer requirement | Technical justification | Trade-offs | Evidence / success criterion |
| --- | --- | --- | --- | --- |
| Use the available L4 | Establish initial feasibility on provisioned compute | Two-step smoke and 100-step pilot training plus export inference have worked on this GPU | Availability supports a POC; throughput, concurrency, memory headroom, and cost are unmeasured | Capture pilot memory/throughput/time and workload constraints before production sizing |
| Use a pinned NGC container and official training recipe | Make the workflow repeatable across engineers | Shared NeMo/PyTorch runtime and NVIDIA implementation preserve the demonstrated path | Requires compatible host driver, staged source/cache, and persistent storage; exact digest not captured | Reproduce setup and restore/evaluate identified artifacts under recorded configuration |
| Keep a small fixed dataset and offline WER policy | Compare recognition fidelity consistently | Same 50 records, prompt, normalization, and corpus scoring isolate checkpoint effects | Cheap development signal, but only 793 words; not representative production or streaming evidence | Compare restored artifacts against 10.84%; agree meaningful quality targets and use untouched representative testing |
| Run a controlled 100-step pilot | Decide whether further GPU investment is justified | More optimization and full internal validation before a larger run | Still small-data adaptation; extra evaluation/checkpoint I/O and overfitting risk | Stable optimization, intended validation events, usable export, measured resources, and interpretable offline quality evidence |
| Separate development, held-out NSC, and GigaSpeech | Avoid selecting a model only because it fits development examples | Freeze choices before in-domain test and external regression checks | More data preparation/evaluation time; corpus distributions do not cover every customer | Demonstrate accepted in-domain quality and acceptable external regressions before recommending rollout |

Customer acceptance thresholds have **not** been agreed. Before production advice, characterize domain terminology, Singapore-English patterns, speaker/recording conditions, streaming latency, real-time factor, throughput/concurrent sessions, and operating cost. Offline WER is useful for transcription fidelity but does not establish these application properties. A numerical improvement on this small development set alone cannot justify a production recommendation.

## Where this guide fits in the agenda

[Guide 01](01-brev-gpu-node-validation.md#agenda-validate-the-stack-from-gpu-to-application) covers host, container, and framework access. This guide continues at **layer 8: application** with the Nemotron ASR POC. NeMo import and model placement extend the reported layer 7 checks. Pretrained GPU inference, an offline baseline on all 50 NSC validation recordings, and an official two-step GPU training smoke test with saved artifacts are verified. Export restoration, fixed-50 fine-tuned evaluation, and baseline reproduction are now verified from October 10 operator evidence. Longer fine-tuning and held-out evaluation remain pending.

### Guide sections

Stage 8 now includes the [completed comparison](#export-evaluation-and-baseline-reproduction--complete), [readiness audit](#pre-training-readiness-review--complete), [completed pilot](#controlled-100-step-pilot--complete), [recording-level comparison](#recording-level-error-comparison--complete), [next investigation](#next-investigate-training-behavior-and-validation-logs--planned), and [customer decision lessons](#lessons-learned-for-customer-decisions).

The [hands-on agenda](#hands-on-agenda) links the ten numbered stages below in the same order as the [README](../../README.md#guide-02--adapting-nvidia-nemotron-35-streaming-asr-to-singapore-english). Use [completion boundaries](#completion-boundaries-and-next-step) for individual milestone status, the [helper-script index](../../scripts/guide-02/README.md) for tooling, and [troubleshooting notes](#troubleshooting-notes) for supporting diagnostics. The [optional one-WAV appendix](#appendix-optional-single-file-inference-smoke-test--not-yet-executed) remains a separate, unexecuted alternative.

## Guide 02 workflow agenda

[Node preflight](#node-preflight--complete), [NeMo/model loading](#nemo-container-and-model-on-gpu--complete), source inspection/separation, annotation auditing, subset generation/validation, [normalization](#transcript-annotation-normalization--complete), and [Nemotron manifest conversion](#nemotron-compatible-manifest-conversion--complete) are complete for the operator-reported checks. Conversion found all 300/50 selected audio files, and one final record per split was inspected. On **October 9, 2026**, the original pretrained model successfully transcribed one recording and then all **50 fixed validation recordings** on the L4. The offline baseline reported **793 reference words and 10.84% WER**. The official two-step training smoke test and checkpoint saving also completed that day. On **October 10, 2026**, stage 8 restored/evaluated the export at **10.97% WER** and reproduced the pretrained **10.84% WER** with the same evaluator. The subsequent configuration audit and metadata eligibility check are complete: **300/300 duration-eligible records / 0.656 h**. The **100-step pilot completed** and its restored export scored **10.47% offline WER**. Recording-level comparison found **9 improved / 7 worsened / 34 equal-error-count recordings**, three net errors fewer than pretrained. Next: investigate training behavior, validation scheduling, data exposure, and checkpoint provenance before more training. Configuration freeze, NSC test, and external evaluation remain pending.

### Current pipeline status

Checkmarks refer to operator-reported checks, including two-step training, export restoration/inference, fixed-50 evaluation, and baseline reproduction. The workflow is validated; smoke did not improve WER, while the 100-step pilot showed preliminary development-set improvement. Full training capacity, further 300-/600-step fine-tuning, held-out evaluation, and streaming performance remain pending.

```text
GPU / container infrastructure ✅
        ↓
Nemotron model load on L4 ✅
        ↓
NSC source data download ✅
        ↓
Source-data inspection ✅
        ↓
Train / validation speaker and ID separation ✅
        ↓
Transcript-tag audit and eligibility analysis ✅
        ↓
POC data policy ✅
        ↓
Deterministic speaker-aware subsets ✅
        ↓
Independent derived-subset validation ✅
        ↓
Transcript annotation normalization ✅
        ↓
Nemotron-compatible manifest conversion ✅
        ↓
Manifest sample inspection ✅
        ↓
Single-utterance pretrained GPU inference ✅
        ↓
Fixed 50-utterance offline validation baseline ✅
793 reference words / 10.84% WER (2026-10-09)
        ↓
Official two-step GPU training ✅
        ↓
Checkpoint artifacts saved ✅
        ↓
Restore exported .nemo and run offline inference ✅
        ↓
Fixed 50-record fine-tuned WER: 10.97% ✅
        ↓
Pretrained baseline recheck: 10.84% ✅
        ↓
Training configuration / duration eligibility audit ✅
300 eligible recordings / 0.656 h
        ↓
100-step pilot training / checkpoint export ✅
        ↓
Pilot export restoration / offline WER: 10.47% ✅
        ↓
Recording-level comparison: 9 improved / 7 worsened / 34 equal errors ✅
← CURRENT CHECKPOINT (2026-10-10)
        ↓
Training behavior / validation / sampling / export investigation ← NEXT — PENDING
        ↓
Longer fine-tuning / validation development loop — PENDING
        ↓
Freeze checkpoint, configuration, and scoring policy — PENDING
        ↓
NSC held-out test — PENDING
        ↓
GigaSpeech external/OOD benchmark — PENDING
        ↓
Streaming latency / performance — PENDING
```

### Hands-on agenda

Follow the same ten stages as the [README overview](../../README.md#guide-02--adapting-nvidia-nemotron-35-streaming-asr-to-singapore-english). These are execution checks; the data-role overview below explains why train, validation, test, and benchmark remain separate. Source/derived-data checks are detailed within stages 4–5.

1. [Node resources](#1-node-resources) — check disk, host RAM, and GPU availability.
2. [NeMo container](#2-nemo-container) — check the pinned Speech container's NeMo version and imports.
3. [Pretrained model](#3-pretrained-model) — restore Nemotron with persistent caching and confirm GPU placement.
4. [Data strategy and source checks](#4-data-strategy-and-source-checks) — define data roles and verify records, split separation, and annotation tags.
5. [Training and validation inputs](#5-training-and-validation-inputs) — build deterministic subsets, normalize text, and verify NeMo audio paths and language fields.
6. [Pretrained inference and baseline](#6-pretrained-inference-and-baseline) — verify one transcription, then establish offline WER on fixed validation.
7. [Training smoke test and artifacts](#7-training-smoke-test-and-artifacts) — run the official GPU smoke test and verify persistent checkpoint files.
8. [Checkpoint comparison and development](#8-checkpoint-comparison-and-development) — restore and compare on fixed validation; iterate longer training and model selection.
9. [Held-out NSC test](#9-held-out-nsc-test) — freeze model, decoding, and scoring choices before evaluating `nsc_test`.
10. [External benchmark](#10-external-benchmark) — check `gigaspeech_test` for generalization/regressions after the NSC test, without routine tuning.

**Current position:** Setup, train/validation preparation, pretrained baseline, two-step training/artifacts, export restoration, fixed-50 comparison, and baseline recheck are complete for the reported scope. **Stage 8 continues:** configuration/duration audits, the 100-step pilot, export evaluation, and recording-level comparison are complete. Training behavior/validation/data exposure/export identity need investigation before any 300- or 600-step experiment. Stage 9's configuration freeze/test and stage 10's external evaluation are planned. Test/benchmark data preparation and overlap checks also remain pending.

Perform data preparation in the SSH-connected GPU host's working directory, outside Git. Download there and bind-mount host data into the container; a laptop path is not automatically available on the remote node. Record source locations/versions, scripts/seed, selected IDs, overlap results, cleanup/scoring rules, container/model versions, and run settings. Keep validation IDs fixed; when scoring rules change, rescore both models consistently. Annotation cleanup, manifest preparation, and the baseline scoring/transcription policy are recorded below. Preserve that policy for comparisons; two-step training settings are recorded below, while longer-run choices and the final model/configuration freeze remain pending.

This POC keeps the experiment small for learning. A customer deployment would choose data coverage and evaluation sizes around actual users, audio conditions, and acceptance criteria; ~300/50 utterances are not a production recommendation. WER checks recognition quality, while streaming behavior, latency, throughput, and scheduler integration remain separate future work.

## 1. Node resources

### Node preflight — complete

**Milestone recorded:** 2026-10-05, the documentation sync date. The following checks were performed on the GPU node before pulling the NeMo container or downloading the model/data. Evidence is the operator's sanitized summary.

```bash
df -h
free -h
nvidia-smi
```

| Resource or check | Observed value |
| --- | --- |
| Root filesystem total | 267 GB, as reported in the milestone summary |
| Root filesystem used | 53 GB |
| Root filesystem available | 214 GB |
| Host RAM total | 15 GiB |
| Host RAM available | Approximately 14 GiB |
| Swap | None configured |
| GPU | NVIDIA L4 |
| GPU VRAM total / used | 23,034 MiB / 0 MiB |
| GPU utilization | 0% |
| NVIDIA driver | `595.91.07` |
| Driver-reported CUDA compatibility level | `13.2` |
| Running GPU processes | None reported |

Disk figures preserve the operator's reported units and rounded values; GNU `df -h` uses powers of 1024 for human-readable sizes. This is one preflight snapshot, not a measurement under model or training load.

#### What the checks establish

- `df -h` checks filesystem capacity and usage; `-h` formats sizes in human-readable units. Check the filesystem where downloads and container storage will live. See the [GNU df manual](https://www.gnu.org/software/coreutils/manual/html_node/df-invocation.html).
- `free -h` checks CPU/system RAM and swap, separately from GPU VRAM. Linux uses otherwise unused RAM for filesystem cache, so low `free` memory can coexist with high `available` memory. `available` estimates memory usable by new applications without swapping, accounting for reclaimable cache. See the [free manual](https://www.man7.org/linux/man-pages/man1/free.1.html).
- `nvidia-smi` checks GPU/driver communication and reports GPU memory, utilization, thermals, power, and active processes. No temperature or power reading was supplied for this milestone. The reported CUDA compatibility level does not establish an installed host Toolkit version. See [NVIDIA's nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Preflight checks help rule out simple causes such as disk exhaustion, current host-memory pressure, or an already-occupied GPU before investigating model/container issues. This snapshot showed available disk and RAM and an idle GPU; it does not yet establish resource sufficiency for the selected model, data, or training configuration.

## 2. NeMo container

<a id="nemo-container-and-model-on-gpu--complete"></a>

### Container version — complete

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator validated the official NeMo Speech container on the Linux GPU host:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo; print(nemo.__version__)"
```

The reported NeMo version was `3.0.0`.

## 3. Pretrained model

### Cached model restoration — complete

The operator successfully restored the model with `ASRModel.from_pretrained(...)` and configured the persistent cache:

```bash
mkdir -p ~/hf-cache

docker run --rm --gpus all \
  -v ~/hf-cache:/root/.cache/huggingface \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo.collections.asr as n; m=n.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b'); m.cuda(); print(next(m.parameters()).device)"
```

The model loaded as `EncDecRNNTBPEModelWithPrompt`. The final output was:

```text
cuda:0
```

This validates Docker GPU passthrough, the NeMo Speech stack for importing ASR and loading this model, and PyTorch/CUDA placement of model parameters on GPU 0. Earlier CUDA-container checks established the separation between the host NVIDIA driver and container CUDA userspace; this step extends that path to Nemotron model loading.

The bind mount stores Hugging Face cache files in `~/hf-cache` on the host, outside the disposable container. Removing the container with `--rm` leaves those files in place for later runs. A second-run cache hit was not separately reported. The model itself fits on the L4 for this loading check. Model placement alone does not establish successful audio inference or sufficient memory for training.

## 4. Data strategy and source checks

### Data strategy and experiment overview

The intended progression is **train → validation/development loop → freeze model/configuration/scoring → NSC held-out test → GigaSpeech external/OOD benchmark**. The **300/50 selected subsets are normalized and converted**, with selected file existence checked and one final record inspected per split. Pretrained inference on all 50 validation recordings is verified, with **10.84% offline WER / 793 reference words**. Two-step smoke training and 100-step pilot training, export restoration, and offline comparison are verified: **10.97%** for smoke, **10.47%** for pilot, versus **10.84%** for the reproduced pretrained baseline. Training exposure and best-export provenance remain open; any 300-/600-step run and final model selection remain pending. Test and benchmark sources are identified in the plan, with local preparation/evaluation pending. **Validation** and **dev** mean the same role; the source directory remains `nsc_dev_3h`.

| Stage | Source → derived POC data | POC / evaluation size | Purpose | Model learns from it? | When used | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| Train | `nsc_query_5h` → `poc/nemo/train_300.jsonl` | 300 selected utterances | Fine-tuning | Yes, directly through gradient updates | First learning stage | ✅ Prepared; two-step smoke and 100-step pilot complete; all 300 duration-eligible, not all confirmed consumed; further training pending investigation |
| Validation | `nsc_dev_3h` → `poc/nemo/dev_50.jsonl` | 50 selected utterances | Baseline comparison, tuning, checkpoint/model decisions | No gradients; influences development indirectly | Before and during fine-tuning development | ✅ Offline pretrained baseline complete: 50 recordings, 793 reference words, 10.84% WER reproduced; smoke export 10.97%, pilot export 10.47%; recording-level analysis complete |
| Test | `nsc_test` | 3,684 utterances / ~7 h, upstream | Final held-out Singapore-English evaluation | No gradients or development tuning | After model/configuration is frozen | Not started; no local download or evaluation documented |
| External benchmark | `gigaspeech_test` | 19,930 utterances / 35.4 h, upstream | Out-of-domain (OOD) generalization/regression check | No gradients or routine tuning | After NSC test evaluation | Not started; no local download or evaluation documented |

Preparation complete here means **subsets checked, normalized/model-facing manifests produced, selected audio-file existence checked, and samples inspected**. Subsequent smoke/baseline runs transcribed the fixed 50-record validation set and established the pretrained offline WER. The official training loader and optimizer path have now run for the two-step smoke test and 100-step pilot; consumption of all 300 recordings and longer training remain unverified. The progression below describes the full intended experiment.

```text
Train: nsc_query_5h → 300-utterance POC subset
     ↓
Gradient updates
     ↓
Validation: nsc_dev_3h → 50-utterance POC subset
     ↓
Development loop: compare models, tune choices, select checkpoint
     ↓
Freeze model and configuration
     ↓
NSC held-out test: nsc_test
     ↓
Final held-out Singapore-English evaluation
     ↓
GigaSpeech external benchmark: gigaspeech_test
     ↓
OOD generalization/regression check
```

Capture baseline **word error rate (WER)** on validation before the first fine-tuning run. Return to the same validation subset to compare checkpoints and development choices. Training and validation can repeat during development; final test evaluation comes after those decisions are settled.

#### Available sources: origin and contents

The [upstream dataset documentation](https://huggingface.co/datasets/pengyizhou/IALP-2026-data) identifies the four sources below. Query/dev counts were recorded locally; test/benchmark sizes are upstream descriptions, not locally verified counts or completed evaluations.

| Source | Origin | Full source size | Contents and intended use |
| --- | --- | --- | --- |
| `nsc_query_5h` | NSC IMDA Part 6 train partition | 2,289 utterances locally; 5.01 h from manifest durations; ~214 MB, recorded as `214M` | FLAC audio and JSONL manifest; 300 selected training utterances used by the two-step route |
| `nsc_dev_3h` | NSC IMDA Part 6 train partition | 1,316 utterances locally; 3.02 h from manifest durations | FLAC audio and JSONL manifest; 50 validation utterances generated for future development/checkpoint decisions |
| `nsc_test` | Official NSC IMDA Part 6 test partition | 3,684 utterances / ~7 h, upstream | FLAC audio and JSONL manifest, per upstream; planned held-out in-domain evaluation |
| `gigaspeech_test` | Separate GigaSpeech test corpus | 19,930 utterances / 35.4 h, upstream | WAV PCM_16 audio and JSONL manifest with normalized references, per upstream; planned external/OOD evaluation |

The downloaded query/dev JSONL records use `id`, `speaker`, `duration` (seconds), `text` (reference transcript), and `audio` (relative path into `audio/`). Their download commands and examples are below. Local train/dev hour totals sum manifest duration fields; test/benchmark hours remain upstream descriptions. Independent checks confirmed 300/50 POC records.

<a id="why-keep-development-data-separate"></a>

#### Direct learning versus development decisions

Training data changes model weights through backpropagation. Validation data must never enter gradient updates, but its results still influence the final system indirectly: an engineer may choose a learning rate, checkpoint, normalization rule, or decoding setting because it improves validation WER. Finalize those choices using validation, and record them before testing.

Test results are for evaluating the finalized internal result. If we repeatedly tune choices based on test results, that set effectively becomes another validation set and loses its role as an independent final check. A new untouched holdout would be needed for a fresh final evaluation. See the [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) for this distinction.

Use query for gradient updates and dev for validation WER, development choices, and checkpoint selection. They share the NSC train partition but have locally verified, separate speaker and utterance-ID sets. Validation influences development, so it does not replace the final `nsc_test` evaluation.

#### What results can tell us

- **Train:** Lower training loss shows better fit to training examples; it does not establish performance on unseen speech.
- **Validation:** WER on the fixed ~50 utterances supports baseline comparisons and development decisions under recorded scoring rules. It is a small, development-influenced result, not an unbiased final estimate for all Singapore English.
- **Test:** `nsc_test` evaluates the frozen system on held-out NSC in-domain speech. Conclusions apply to this test distribution; they do not establish production performance or generalization to every customer.
- **External benchmark:** `gigaspeech_test` checks generalization and regressions on a separate corpus. Compare baseline and final-model results under fixed scoring rules, without routine tuning on this benchmark. Its results do not establish universal generalization.

#### Evaluation preparation — TODO

- **NSC test:** Prepare `nsc_test` on the GPU host, verify local counts and overlap, and keep it out of the development loop. Evaluate only after the model, normalization, and decoding configuration are frozen.
- **GigaSpeech OOD benchmark:** Prepare `gigaspeech_test` on the host and document local counts and scoring/reference normalization. Run the planned generalization/regression comparison after NSC test evaluation. Preparation and both evaluations remain unvalidated.

### NSC training/query and development data — downloaded and extracted

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator downloaded the small NSC Part 6 training/query dataset and the separate development split from [IALP-2026-data on Hugging Face](https://huggingface.co/datasets/pengyizhou/IALP-2026-data). The counts and records below are operator-reported observations. Keep downloaded archives and audio on the GPU host, outside the repository.

For reproduction, use a host data directory outside the Git checkout. Both archives extract into the current directory. The commands below use the same layout as the later Brev manifest checks (`/home/ubuntu/data/nsc` for the recorded `ubuntu` account):

```bash
mkdir -p ~/data/nsc
cd ~/data/nsc
```

#### Training/query split

```bash
wget https://huggingface.co/datasets/pengyizhou/IALP-2026-data/resolve/main/nsc-query.tar.gz
tar xzf nsc-query.tar.gz
wc -l nsc_query_5h/manifest.jsonl
du -sh nsc_query_5h
```

Reported results: **2,289 manifest records** and **214M** extracted directory size (`du -sh`). The source contains approximately **5 hours** of speech; its duration was not independently summed in this milestone.

```text
nsc_query_5h/
├── audio/
├── manifest.jsonl
├── text
├── utt2spk
└── wav.scp
```

Example training record (ID and audio filename shortened):

```json
{"id":"...","speaker":"00038","duration":2.58,"text":"call one telco","audio":"audio/...flac"}
```

#### Separate development split

```bash
wget https://huggingface.co/datasets/pengyizhou/IALP-2026-data/resolve/main/nsc-dev.tar.gz
tar xzf nsc-dev.tar.gz
wc -l nsc_dev_3h/manifest.jsonl
```

Reported result: **1,316 manifest records**, with approximately **3 hours** of speech in the source. Its duration was not independently summed in this milestone. This is our validation source, not an already selected final test set.

Example development record (ID and audio filename shortened):

```json
{"id":"...","speaker":"00017","duration":6.24,"text":"okay sure <v-noise> uh good afternoon may i have your contact number in case the line like get uh disconnected","audio":"audio/...flac"}
```

The inspected train/validation records share the fields `id`, `speaker`, `duration`, `text`, and `audio`, with relative FLAC paths. The original manifests remain in source format; model-facing derived copies are produced in the [conversion section](#nemotron-compatible-manifest-conversion--complete).

### Source manifest checks — complete

The [helper-script index](../../scripts/guide-02/README.md) lists each tool's inputs and whether it writes output. These checks inspect manifest records. Full-source audio integrity remains unaudited. Later conversion checks selected-file existence, and separate inference/baseline milestones process the 50 selected validation recordings. This does not audit the full source corpus or training audio.

Current original manifests on the Brev host, unchanged:

```text
/home/ubuntu/data/nsc/nsc_query_5h/manifest.jsonl
/home/ubuntu/data/nsc/nsc_dev_3h/manifest.jsonl
```

#### Split construction and hands-on verification

`nsc_query_5h` and `nsc_dev_3h` are separate selections from the NSC Part 6 train partition, constructed using different speaker sets. **`nsc_dev_3h` is not a subset of `nsc_query_5h`**. The [publisher's split-construction notes](https://huggingface.co/datasets/pengyizhou/IALP-2026-data#split-construction-nsc) state that query, dev, and `nsc_test` are mutually speaker-disjoint, and query/dev exclude every speaker appearing in the official NSC test partition. `nsc_test` comes from that official test partition.

**Verified on the Brev node:** The operator used [`check_split_overlap.py`](../../scripts/guide-02/check_split_overlap.py) to check the complete train and validation manifests. The script accepts both manifest paths at runtime and only reads them.

Rerun from the repository root **on the GPU compute node**. The checkout and scripts must exist on that node; a laptop checkout is not automatically available over SSH. These reproduction commands map the host source directory to `/data/nsc` inside the container; change the mount and arguments if your data lives elsewhere.

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v "$PWD/scripts/guide-02:/scripts:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /scripts/check_split_overlap.py \
  /data/nsc/nsc_query_5h/manifest.jsonl \
  /data/nsc/nsc_dev_3h/manifest.jsonl
```

Verified Brev output:

```text
Train records:   2289
Dev records:     1316
Train speakers:  111
Dev speakers:    64
Speaker overlap: 0
ID overlap:      0
```

The complete train and validation manifests share no speakers or utterance IDs; dev is not simply a subset of query. This independently confirms the upstream train/dev speaker-disjoint design. It does **not** verify separation from `nsc_test`: compare query/dev/test pairwise when preparing test. The derived POC subsets were also [checked independently](#derived-poc-subset-validation--complete).

**SA/reproducibility:** Both checks ran inside the version-pinned NeMo Speech container with the source dataset mounted read-only. In these commands, `:ro` makes the data and script mounts read-only; `/scripts` is the container's script directory, and manifest paths are command-line arguments. Python comes from the container, without installing it on the host or modifying the dataset. Neither helper contains Brev-specific hardcoded paths. Keep the source version/location and script version with the run notes; the download commands above are reproduction examples.

<a id="transcript-annotations-and-normalization--planned"></a>

#### Transcript annotation audit — complete

**Verified observation:** The operator inspected both complete source manifests for `<...>` annotation tokens using [`inspect_transcript_tags.py`](../../scripts/guide-02/inspect_transcript_tags.py). It accepts one or more manifest paths, scans transcript text, and reports tagged records separately from individual token occurrences.

From the same repository root on the compute node:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v "$PWD/scripts/guide-02:/scripts:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /scripts/inspect_transcript_tags.py \
  /data/nsc/nsc_query_5h/manifest.jsonl \
  /data/nsc/nsc_dev_3h/manifest.jsonl
```

Verified Brev results, summarized by source (the script also prints each manifest path):

```text
Train: nsc_query_5h
Total records:      2289
Records with tags:   511
<v-noise>:           598
<unk>:               145
<noise>:              49
```

```text
Validation: nsc_dev_3h
Total records:      1316
Records with tags:   286
<v-noise>:           345
<unk>:               104
<noise>:              24
```

Tags are common in both sources. Token counts are occurrences, not distinct records; a transcript can contain several tags. The reported audit found these three types in the checked manifests, rather than only `<v-noise>`.

| Token | Intended NSC annotation meaning |
| --- | --- |
| `<v-noise>` | Vocal/non-lexical noise annotation, not an ordinary spoken word |
| `<noise>` | Non-vocal/background noise annotation |
| `<unk>` | Unclear/unidentified speech under the NSC transcription convention |

**Data-quality caveat:** The intended definition does not guarantee that every individual `<unk>` label is correct. Transcript context suggests that some occurrences could represent identifiable Singapore-English/Singlish speech, such as discourse particles or locally accented speech. This is a hypothesis: **we have not listened to the source audio to verify it**. It does not establish that `<unk>` means Singlish or that the annotations are wrong.

**Current POC decision:** We will not manually listen to, relabel, or correct individual `<unk>` occurrences in this tutorial. Such corrections could introduce subjective labels and reduce reproducibility; the scope remains a simple infrastructure/fine-tuning POC. Preserve the original source manifests unchanged.

**Future data-quality work:** Sample utterances containing `<unk>` → listen to source audio → categorize genuinely unintelligible speech versus possible transcription errors → optionally create a reviewed/corrected derived dataset. This work has not been performed.

#### POC eligibility impact — complete

The operator validated [`check_poc_eligibility.py`](../../scripts/guide-02/check_poc_eligibility.py) to measure what would remain if whole utterances containing `<unk>` were excluded. It only reads manifests; it does not create a filtered dataset.

Exact command used from the Brev host:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/check_poc_eligibility.py \
    /data/nsc/nsc_query_5h/manifest.jsonl \
    /data/nsc/nsc_dev_3h/manifest.jsonl
```

`/home/ubuntu/data/nsc` is mounted read-only as `/data/nsc`; `/home/ubuntu/work/nemotron-poc` is mounted read-only as `/work`. The script was present in that host work directory for this run; its repository location is `scripts/guide-02/`. Manifest paths are supplied at runtime, with no hardcoded host paths in the script. No GPU access is required for this CPU/data-integrity task.

Observed output:

```text
Manifest: /data/nsc/nsc_query_5h/manifest.jsonl
Total records:       2289
Records with <unk>:  131
Records remaining:   2158
Total duration:      5.01 hours
<unk> duration:      0.42 hours
Remaining duration:  4.59 hours

Manifest: /data/nsc/nsc_dev_3h/manifest.jsonl
Total records:       1316
Records with <unk>:  91
Records remaining:   1225
Total duration:      3.02 hours
<unk> duration:      0.31 hours
Remaining duration:  2.71 hours
```

These figures describe the eligible source pools; the read-only check creates no filtered files. Subsequent subset generation is documented below. The check counts each `<unk>` utterance once, explaining why 131/91 affected records differ from the earlier 145/104 token occurrences.

##### POC annotation policy — decision complete

| Item | Current POC policy |
| --- | --- |
| `<unk>` | Exclude the whole utterance from the train/validation POC |
| `<v-noise>`, `<noise>` | Keep the utterance; remove annotation tokens during the completed normalization step |
| Local speech such as `lah`, `wah`, `ya`, `mm` | Keep as normal speech |
| Original manifests | Never modify; write derived data separately |

Excluding `<unk>` is a pragmatic tutorial choice, not a claim that these utterances are bad data. The tag indicates speech was not confidently transcribed; deleting only the token could leave spoken content without corresponding text and create audio/text misalignment. We are deliberately avoiding manual relabeling. The eligible **2,158 train records / 4.59 h** and **1,225 validation records / 2.71 h** comfortably exceed the chosen 300/50 POC sizes.

The earlier manual-audio-review note remains future work for a more rigorous data-quality project. No individual annotations are corrected in this POC. The generator below excludes `<unk>` utterances; the later normalization script removes only the two approved noise-tag types.

<a id="transcript-normalization--planned"></a>

The [completed normalization step](#transcript-annotation-normalization--complete) applies this policy to separate derived files, preserving the original manifests. Annotation cleanup remains distinct from final WER-scoring choices.

<a id="next-create-tiny-deterministic-poc-subsets--planned"></a>
<a id="next-data-preparation-on-the-compute-node--planned"></a>

## 5. Training and validation inputs

### Deterministic POC subset generation — complete

The operator successfully ran [`create_poc_subsets.py`](../../scripts/guide-02/create_poc_subsets.py) on the Brev node. It reads source manifests, excludes whole `<unk>` utterances, and writes selected source-format records to a separate output directory. Transcripts are not normalized or converted to NeMo format by this script.

#### Design decisions

The eligible pools were **2,158 train utterances / 4.59 h** and **1,225 validation utterances / 2.71 h** after excluding `<unk>`. We chose **300 train / 50 validation utterances** for fast end-to-end learning on the NVIDIA L4. This is a POC engineering choice, not a production-scale train/validation ratio.

- **Train:** Cycle across speakers, taking at most one utterance per speaker per pass, to avoid a few prolific speakers dominating the small subset.
- **Validation:** Select one utterance from each of 50 distinct speakers to maximize speaker breadth.
- **Determinism:** `seed = 42`; train uses `random.Random(seed)` and validation uses `random.Random(seed + 1)`. Separate RNG streams keep changes to one subset size from unexpectedly changing the other. Reproduction assumes the same source manifests and ordering.

#### Derived-data directory and container execution

The original manifests remain unchanged. The operator created a dedicated derived-data directory on the host:

```bash
mkdir -p ~/data/nsc/poc
```

```text
/home/ubuntu/data/nsc/
├── nsc_query_5h/
│   ├── manifest.jsonl                 # original training source
│   └── audio/                         # original training FLAC files
├── nsc_dev_3h/
│   ├── manifest.jsonl                 # original validation source
│   └── audio/                         # original validation FLAC files
└── poc/
    ├── train_300.jsonl                # selected source-format records
    ├── dev_50.jsonl
    ├── normalized/
    │   ├── train_300.jsonl            # noise annotations removed from text
    │   └── dev_50.jsonl
    └── nemo/
        ├── train_300.jsonl            # model-facing manifest
        └── dev_50.jsonl
```

Current lineage: **source manifest → filter/sample → POC subset → transcript-text cleanup → normalized manifest → resolve paths/map fields → Nemotron manifest**. All earlier stages remain. Source audio stays in its original directories; neither new script modifies it. Executed scripts live separately in `/home/ubuntu/work/nemotron-poc/`, with version-controlled copies under `scripts/guide-02/`; committing them does not deploy them to Brev.

Successful command from the Brev host:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/data/nsc/poc:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/create_poc_subsets.py \
    /data/nsc/nsc_query_5h/manifest.jsonl \
    /data/nsc/nsc_dev_3h/manifest.jsonl \
    /output \
    --train-count 300 \
    --dev-count 50 \
    --seed 42
```

| Host path | Container path | Access |
| --- | --- | --- |
| `/home/ubuntu/data/nsc` | `/data/nsc` | Read-only source data (`:ro`) |
| `/home/ubuntu/work/nemotron-poc` | `/work` | Read-only tooling (`:ro`) |
| `/home/ubuntu/data/nsc/poc` | `/output` | Read/write derived output (`:rw`) |

The only writable host bind mount is the dedicated derived-data directory. The script was present in the host work directory for this run; its repository location is `scripts/guide-02/`. Manifest and output paths are runtime arguments. No `--gpus all` is required because generation is a CPU/data task.

#### Observed successful output

The generator reported:

```text
Train output:    /output/train_300.jsonl
Train records:   300
Train speakers:  111
Train duration:  0.66 hours

Dev output:      /output/dev_50.jsonl
Dev records:     50
Dev speakers:    50
Dev duration:    0.11 hours
```

Train: **300 utterances across all 111 eligible train speakers**, approximately **0.66 h / 40 minutes**. Validation: **50 utterances from 50 distinct speakers**, approximately **0.11 h / 6.6 minutes**. Durations sum manifest fields and are rounded. Record and speaker counts were independently checked below.

<a id="next-validate-derived-subsets--pending"></a>

### Derived POC subset validation — complete

The operator independently checked the generated manifests on the Brev node, rather than relying on the generator's report. These checks cover file ownership, record counts, annotation tags, speakers, and utterance-ID separation; they do not establish audio integrity or model performance.

#### Host-side file ownership and line counts

```bash
ls -lh ~/data/nsc/poc
```

Initial ownership in this run:

```text
-rw-r--r-- 1 root root 13K ... dev_50.jsonl
-rw-r--r-- 1 root root 82K ... train_300.jsonl
```

The container's default execution context produced root-owned output. Ownership was corrected **only for the two derived manifests**, then checked again with `ls -lh`; no original NSC files were changed:

```bash
sudo chown ubuntu:ubuntu \
  ~/data/nsc/poc/train_300.jsonl \
  ~/data/nsc/poc/dev_50.jsonl
```

Verified ownership:

```text
-rw-r--r-- 1 ubuntu ubuntu 13K ... dev_50.jsonl
-rw-r--r-- 1 ubuntu ubuntu 82K ... train_300.jsonl
```

Independent line-count check:

```bash
wc -l \
  ~/data/nsc/poc/train_300.jsonl \
  ~/data/nsc/poc/dev_50.jsonl
```

Observed:

```text
300 /home/ubuntu/data/nsc/poc/train_300.jsonl
 50 /home/ubuntu/data/nsc/poc/dev_50.jsonl
350 total
```

The tag and overlap helpers below also parsed all 300/50 records as JSONL.

#### Derived annotation tags

The existing [`inspect_transcript_tags.py`](../../scripts/guide-02/inspect_transcript_tags.py) checked both derived files. To reproduce on the compute node with the helper present in the host work directory:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/inspect_transcript_tags.py \
    /data/poc/train_300.jsonl \
    /data/poc/dev_50.jsonl
```

Observed results:

```text
Manifest: /data/poc/train_300.jsonl
Total records:       300
Records with tags:    61
Tag counts:
  <v-noise>: 87
  <noise>: 5

Manifest: /data/poc/dev_50.jsonl
Total records:       50
Records with tags:    4
Tag counts:
  <v-noise>: 7
```

**`<unk>`: 0 in both subsets.** The whole-utterance exclusion worked. `<v-noise>` and sampled `<noise>` remain for later token removal; this is not normalized output.

#### Derived speaker and utterance-ID separation

The existing [`check_split_overlap.py`](../../scripts/guide-02/check_split_overlap.py) independently checked the POC pair. Reproduction command:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/check_split_overlap.py \
    /data/poc/train_300.jsonl \
    /data/poc/dev_50.jsonl
```

Observed:

```text
Train records:   300
Dev records:     50
Train speakers:  111
Dev speakers:    50
Speaker overlap: 0
ID overlap:      0
```

All 111 eligible train speakers are represented; validation contains 50 distinct speakers. The POC manifests share no speakers or utterance IDs. Both helpers use the pinned NeMo image with data/tooling mounted read-only and no GPU request; the commands above are reproduction examples. Source and derived checks remain separate evidence.

<a id="current-verified-poc-datasets"></a>

#### Verified source-format POC subsets

| Role | Manifest on the Brev host | Utterances | Speakers | Duration | `<unk>` |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | `/home/ubuntu/data/nsc/poc/train_300.jsonl` | 300 | 111 | ~0.66 h | 0 |
| Validation | `/home/ubuntu/data/nsc/poc/dev_50.jsonl` | 50 | 50 | ~0.11 h | 0 |

Speaker overlap: **0**. Utterance-ID overlap: **0**. These host paths map to container paths through the bind mounts above. Noise tokens remain in these preserved source-format subsets; the normalized copies below contain the cleaned transcripts.

<a id="next-transcript-normalization--planned"></a>

### Transcript annotation normalization — complete

**Milestone recorded:** 2026-10-08 documentation sync. The commands and outputs in this section and the conversion section are the operator's reported Brev-node results, not commands independently executed by the guide maintainer.

#### Implemented policy and host script

The operator created [`normalize_transcripts.py`](../../scripts/guide-02/normalize_transcripts.py) in the SSH-connected host workspace:

```bash
cd ~/work/nemotron-poc
vim normalize_transcripts.py
```

Its repository copy lives under `scripts/guide-02/`; the executed copy lives under `/home/ubuntu/work/nemotron-poc/`. A GitHub commit does not deploy it to Brev, and a checkout on the node is not confirmed. See the [tooling index](../../scripts/guide-02/README.md#scripts-and-host-paths) for this distinction.

The script replaces only `<v-noise>` and `<noise>` with spaces, collapses whitespace, and trims the transcript. It preserves actual words, identifiable Singlish particles, fillers, case, punctuation, and every other record field (`id`, `speaker`, `duration`, `audio`, and other metadata). It stops on unexpected `<...>` tags, including a surviving `<unk>`, or an empty cleaned transcript; it does not silently drop selected utterances.

```python
cleaned = cleaned.replace(tag, " ")
```

A space prevents adjacent words from joining.

```python
cleaned = " ".join(cleaned.split())
```

Whitespace cleanup does not rewrite the words. Whole `<unk>` utterances were excluded during subset selection; normalization does not delete unknown speech from the audio. This is **annotation cleanup**, not audio denoising or a complete WER-scoring policy.

```python
destination.open("x", encoding="utf-8")
```

Exclusive creation protects an existing destination. A rerun to that path should fail; preserve validated artifacts rather than deleting them blindly.

#### Normalization execution and reported output

Created the separate output directory:

```bash
mkdir -p ~/data/nsc/poc/normalized
```

**Train:**

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/data/nsc/poc/normalized:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/normalize_transcripts.py \
    /data/poc/train_300.jsonl \
    /output/train_300.jsonl
```

```text
Input:             /data/poc/train_300.jsonl
Output:            /output/train_300.jsonl
Records written:   300
Text changed:      61
Removed <v-noise>: 87
Removed <noise>: 5
```

**Validation:**

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/data/nsc/poc/normalized:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/normalize_transcripts.py \
    /data/poc/dev_50.jsonl \
    /output/dev_50.jsonl
```

```text
Input:             /data/poc/dev_50.jsonl
Output:            /output/dev_50.jsonl
Records written:   50
Text changed:      4
Removed <v-noise>: 7
Removed <noise>: 0
```

The reported 61/4 changed transcripts and removal counts (train: 87 `<v-noise>`, 5 `<noise>`; validation: 7/0) agree with the earlier POC inventory. Inputs and tooling were read-only; only the normalized-output host mount was writable. No audio was copied, resampled, denoised, or otherwise modified.

A later independent tag-inspection attempt failed because the helper was missing from the runtime workspace; it did not produce a successful audit. The completed evidence is these normalization reports and their agreement with the earlier inventory. Conversion below subsequently parsed every normalized record. See the [deployment note](#missing-runtime-helper) for the failed check; no suggested `grep` check is claimed complete.

### Nemotron-compatible manifest conversion — complete

#### Recipe choice: five fields

Before finalizing the converter, the operator reviewed [NVIDIA's fine-tuning article](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-for-saudi-arabic-dialects-with-a-path-to-other-languages/) and its [prompt-aware notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb). The notebook emits the usual three ASR fields plus `lang` and `target_lang`. We chose to follow that five-field format for this Nemotron recipe; it is not a universal requirement for every NeMo model or loader.

| Field | Meaning in our prepared manifest |
| --- | --- |
| `audio_filepath` | Absolute audio-file path visible inside the runtime container |
| `duration` | Seconds carried over from source metadata |
| `text` | Normalized reference transcript |
| `lang` | Language metadata included to follow the example recipe |
| `target_lang` | Target-language metadata for the prompt-aware recipe |

Both language fields are `en-US`, following the notebook. The [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#supported-languages) lists `en-US` and `en-GB` as supported English locales. This choice does not relabel Singapore recordings as American speech or establish an optimal prompt; we do not invent `en-SG`. These are metadata fields, not a manually appended `<en-US>` transcript token. The smoke script below reads `target_lang` from the first record into an explicit transcription configuration. The five-field format was subsequently exercised by the official two-step training route below; this does not establish consumption of every training recording.

#### Audio paths and provenance

Source-format records contain a relative path:

```json
{
  "audio": "audio/example.flac"
}
```

For train, `audio_root = /data/nsc/nsc_query_5h` plus `audio/example.flac` resolves to `/data/nsc/nsc_query_5h/audio/example.flac`. Validation uses `/data/nsc/nsc_dev_3h`. The root is the original dataset directory, **not its `audio/` subdirectory**, because the relative field already includes `audio/`. Do not resolve against the normalized-manifest directory.

These absolute paths are **container paths**. Future inference/training must preserve the `/data/nsc` mount layout or regenerate manifests for another layout. `/output` determines the manifest's destination; embedded audio paths still point to `/data/nsc/...`.

The executed converter writes only the five fields above, omitting `id` and `speaker`. Those remain in the POC/normalized manifests for provenance and speaker checks. Do not run `check_split_overlap.py` directly on the final five-field files: that helper expects `id` and `speaker`.

#### Conversion execution and reported output

Created [`convert_to_nemo_manifest.py`](../../scripts/guide-02/convert_to_nemo_manifest.py) on the host and its output directory:

```bash
cd ~/work/nemotron-poc
vim convert_to_nemo_manifest.py
```

```bash
mkdir -p ~/data/nsc/poc/nemo
```

**Train:**

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/data/nsc/poc/nemo:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/convert_to_nemo_manifest.py \
    /data/nsc/poc/normalized/train_300.jsonl \
    /data/nsc/nsc_query_5h \
    /output/train_300.jsonl \
    --language en-US
```

```text
Input:           /data/nsc/poc/normalized/train_300.jsonl
Output:          /output/train_300.jsonl
Records written: 300
Language:        en-US
Audio paths:     verified
```

**Validation:**

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/data/nsc/poc/nemo:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/convert_to_nemo_manifest.py \
    /data/nsc/poc/normalized/dev_50.jsonl \
    /data/nsc/nsc_dev_3h \
    /output/dev_50.jsonl \
    --language en-US
```

```text
Input:           /data/nsc/poc/normalized/dev_50.jsonl
Output:          /output/dev_50.jsonl
Records written: 50
Language:        en-US
Audio paths:     verified
```

The three positional arguments are `input_manifest` (normalized records), `audio_root` (original dataset directory), and `output_manifest` (new NeMo JSONL). `--language en-US` fills both language fields. Like normalization, conversion refuses to overwrite an existing output.

| Host path | Container path | Access |
| --- | --- | --- |
| `/home/ubuntu/data/nsc` | `/data/nsc` | Read-only |
| `/home/ubuntu/work/nemotron-poc` | `/work` | Read-only |
| `/home/ubuntu/data/nsc/poc/nemo` | `/output` | Read/write |

Both preparation steps used the same version-tagged NeMo Speech image; no image digest was recorded. Neither requested a GPU, and Python ran in the container without a host Python installation. The image's default execution context was retained after the earlier UID issue. **Ownership checks/corrections for `normalized/` and `nemo/` were not reported**; earlier ownership correction covered only the two original POC subset files.

**What conversion establishes:** It produced 300/50 records and checked that every embedded audio path resolves to an existing file within the supplied source root. It does not decode FLAC, check sample rate/channels, remeasure durations, establish transcription accuracy or loader compatibility, or prove training fits in L4 memory. Durations are copied from metadata.

#### Output-manifest sample inspection — complete

The operator printed one record from each JSONL file:

```bash
head -n 1 ~/data/nsc/poc/nemo/train_300.jsonl
head -n 1 ~/data/nsc/poc/nemo/dev_50.jsonl
```

`head -n 1` prints the first line, one record here. Actual train record:

```json
{"audio_filepath": "/data/nsc/nsc_query_5h/audio/imda-2021-part6-10105-channel001m-0047676-0048681.flac", "duration": 10.05, "text": "uh do you have a preferred timing two p m let me check whether the two p m slot is available", "lang": "en-US", "target_lang": "en-US"}
```

Actual validation record:

```json
{"audio_filepath": "/data/nsc/nsc_dev_3h/audio/imda-2021-part6-07552-channel001m-0019035-0019311.flac", "duration": 2.76, "text": "uh correct", "lang": "en-US", "target_lang": "en-US"}
```

These observed examples show the chosen five fields, correct source directories, and preserved spoken fillers such as `uh`. This is sample inspection, not a complete independent audit of every output field. **Manifest conversion complete** means files produced, all selected file paths found, and these two records inspected; conversion itself does not establish a NeMo training batch, audio decoding, inference, or training success. The separate single-record inference result is below.

<a id="next-single-utterance-gpu-inference--planned"></a>

## 6. Pretrained inference and baseline

### Single-utterance pretrained GPU inference — complete

**Verified on October 9, 2026:** The operator reported a successful Brev-node run on the **NVIDIA L4 24 GB**, using `nvcr.io/nvidia/nemo-speech:26.07.00` and the original `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint. NeMo logs explicitly confirmed model restoration as `EncDecRNNTBPEModelWithPrompt`. These are reported node results, not a GPU run independently executed by the guide maintainer.

The script at `/home/ubuntu/work/nemotron-poc/baseline_smoke.py` reads the first recording and reference from the prepared **50-record** NSC Part 6 validation manifest. Only **one recording** was transcribed. The [repository script](../../scripts/guide-02/baseline_smoke.py) implements the reported workflow and full successful configuration with explanatory comments; its bytes have not been compared with the Brev file. Repository and runtime copies remain separate locations.

#### Initial prompt failure and successful workaround

The initial transcription attempt passed `target_lang="en-US"` to `model.transcribe()` but encountered a missing language prompt:

```text
ValueError: Unknown prompt key: 'None'
```

The successful run supplied an explicit prompt-aware transcription configuration:

```python
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
)

transcribe_cfg = RNNTPromptTranscribeConfig(
    use_lhotse=False,
    batch_size=1,
    num_workers=0,
    target_lang=target_lang,
    verbose=False,
)
with torch.inference_mode():
    results = model.transcribe(
        audio=[audio_path],
        override_config=transcribe_cfg,
    )
```

`target_lang` comes from the manifest (`en-US` here). `use_lhotse=False` selects the non-Lhotse transcription path; `batch_size=1` and `num_workers=0` keep this one-file check simple, and `verbose=False` suppresses transcription progress output. Disabling Lhotse **together with the explicit configuration** resolved the observed error. This is a successful workaround, not definitive proof of the internal root cause or evidence that Lhotse is generally unsuitable.

The script restores the pretrained model, calls `model.cuda()` and `model.eval()`, then performs transcription inside `torch.inference_mode()`. Evaluation mode selects evaluation behavior; inference mode disables gradient tracking. It prints raw reference/prediction text and computes no WER.

#### Executed container command and observed result

The operator ran:

```bash
docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_smoke.py /data/nsc/poc/nemo/dev_50.jsonl
```

`--gpus all` exposes the GPU; the cache mount persists model files outside the disposable container. The dataset mount preserves the manifests' `/data/nsc/...` audio paths and is read-only, protecting original customer data. The read-only `/work` mount supplies the host script. `--rm` removes the stopped container while host cache/data/scripts persist. Run from the SSH-connected Brev host; laptop paths are not automatically present there.

Reported output:

```text
Model class : EncDecRNNTBPEModelWithPrompt
Model device: cuda:0

Running inference...

=== RESULTS ===
Reference : uh correct
Prediction: Uh correct. <en-US>

Smoke test completed.
```

This verifies that the restored pretrained model could process this validation recording and produce text on GPU 0 through the containerized stack. Model restoration alone had not established inference. The result does not establish all-50-record decoding, training-loader compatibility, fine-tuning, true streaming, latency, throughput, runtime, or memory consumption. No WER was calculated.

#### SA lessons and scoring handoff

- Multilingual Nemotron needs appropriate language conditioning; retain the actual prompt/configuration when reproducing a failure. Framework default data-loading settings may need adjustment for the tested execution path.
- Raw predictions may include a language marker such as `<en-US>`. **Exclude language markers from WER scoring while keeping raw predictions available.** The [NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#streaming-inference) describes language conditioning and tag handling.
- Use identical text-normalization and decoding policies for pretrained/fine-tuned checkpoint comparisons. This output also differs from the reference in case and punctuation; the baseline below records how those differences were normalized for scoring.
- Read-only data mounts protect source data while troubleshooting application behavior.

<a id="next-50-utterance-validation-baseline-and-wer--planned"></a>

### Pretrained 50-utterance validation baseline — complete

**Verified on October 9, 2026:** The operator successfully ran the original pretrained `nvidia/nemotron-3.5-asr-streaming-0.6b` model on **all 50 fixed NSC Part 6 validation recordings** using the NVIDIA L4 24 GB Brev instance and `nvcr.io/nvidia/nemo-speech:26.07.00`. These are operator-reported node results; the guide maintainer did not independently execute or recalculate this evaluation.

#### Executed command and reported output

```bash
docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  -v "$HOME/work/nemotron-poc/results/baseline:/results" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --output-dir /results
```

```text
Model class : EncDecRNNTBPEModelWithPrompt
Model device: cuda:0

Transcribing 50 recordings...

=== BASELINE RESULTS ===
Utterances     : 50
Reference words: 793
WER            : 10.84%

=== SAVED FILES ===
/results/baseline_predictions.jsonl
/results/baseline_metrics.json
```

The dataset and script mounts are read-only. The added writable `/results` mount maps to `/home/ubuntu/work/nemotron-poc/results/baseline` on Brev, so the prediction/metric files survive container removal. The Hugging Face cache also remains on the host. Run the command on the SSH-connected compute node; it uses the evaluator already present there, not a laptop path.

#### Reported evaluation implementation and scoring policy

`/home/ubuntu/work/nemotron-poc/baseline_eval.py` reads the fixed JSONL manifest, restores the original pretrained checkpoint, transcribes its recordings, applies identical normalization to reference/prediction text, and calculates dataset-level WER using NeMo's `word_error_rate()`. It saves raw and normalized predictions plus evaluation metadata.

The successful transcription configuration was:

```python
transcribe_cfg = RNNTPromptTranscribeConfig(
    use_lhotse=False,
    batch_size=1,
    num_workers=0,
    target_lang="en-US",
    verbose=False,
)
```

The reported scoring normalization:

- Lowercases text.
- Removes language markers such as `<en-US>`.
- Normalizes curly apostrophes.
- Removes punctuation other than apostrophes.
- Collapses whitespace.

Apply these same rules to references and predictions, and reuse the same transcription/decoding policy for pretrained/fine-tuned comparisons. Annotation cleanup during data preparation and text normalization for scoring are separate transformations.

[NeMo's WER implementation](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/metrics/wer.py) aggregates word-level edit errors over total reference words; this is dataset-level WER, not an unweighted average of utterance percentages. **10.84%** is the reported displayed result over **793 reference words**; no per-utterance scores or error-category counts are provided here.

**Evaluator-source status (October 10):** The operator supplied the complete **238-line [baseline_eval.py](../../scripts/guide-02/baseline_eval.py)** during this repository update. It is added unchanged, with the established pretrained/local-export interface, normalization, corpus-level WER, raw/normalized outputs, and metric metadata. Local syntax and stubbed functional checks do not constitute independent NeMo/GPU execution or a direct byte comparison with the Brev filesystem. No source reconstruction or scoring change was needed.

#### Interpretation and portfolio lessons

The operator observed that the model sometimes changes spoken nonstandard grammatical constructions into more standard English. A result can read more naturally while introducing WER substitutions against the spoken reference. This is a qualitative observation; no example records or quantified error breakdown are added. Preserve the intended Singapore-English references rather than silently rewriting them to match the model.

- **Fixed validation set:** The same recordings make changes attributable to the model/configuration rather than a different sample.
- **Consistent normalization:** Score both checkpoints under identical rules; if rules change, rescore both.
- **Original pretrained baseline:** Establishes the starting point before gradient updates, so later fine-tuning gains or regressions have a comparison.
- **Raw predictions:** Preserve capitalization, punctuation, language markers, and model wording for error inspection; retain normalized versions for scoring.

This is a **small development-set result from offline transcription**, not a production-quality benchmark, final held-out result, or streaming WER. This baseline alone does not establish training or checkpoint-comparison results; no latency, throughput, runtime, or memory-consumption measurement is claimed.

Keep original NSC audio, source dataset contents, and transcript-bearing raw/normalized prediction files outside Git. This guide and progress log retain aggregate metrics and the reproduction command; the reported output files remain in the host results directory.

## 7. Training smoke test and artifacts

<a id="next-training-smoke-test--planned"></a>

### Official NeMo Fine-Tuning — Two-Step GPU Smoke Test

**Verified execution: October 9, 2026; documentation synced October 10.** The operator completed two optimizer steps on the Brev Ubuntu node's **NVIDIA L4 24 GB**, using `nvcr.io/nvidia/nemo-speech:26.07.00` (NeMo Speech 3.0.0). The original pretrained `nvidia/nemotron-3.5-asr-streaming-0.6b` was restored as `EncDecRNNTBPEModelWithPrompt`. Training, limited validation, and checkpoint/model export succeeded. Evidence is the supplied operator command/log history and host artifact listing; the guide maintainer did not independently run GPU training.

The objective was **training functionality**, not accuracy optimization or proof of full training capacity. Subsequent export restoration and comparable post-training WER are complete in [stage 8](#8-checkpoint-comparison-and-development); the subsequent 100-step pilot is recorded in stage 8; further 300-/600-step training remains pending.

#### Official source and experiment architecture

Use [NeMo Speech **v3.0.0**](https://github.com/NVIDIA-NeMo/Speech/tree/v3.0.0), its [`speech_to_text_finetune.py` entry point](https://github.com/NVIDIA-NeMo/Speech/blob/v3.0.0/examples/asr/speech_to_text_finetune.py), and the [prompt-aware streaming YAML](https://github.com/NVIDIA-NeMo/Speech/blob/v3.0.0/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml). The [NVIDIA fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb) provides recipe context. Its example uses a moving branch and different training settings; this experiment pins the Speech source tag and uses the overrides below.

`/home/ubuntu/work/nemotron-poc/train_smoke.py` was an earlier custom-loop proposal, **superseded before execution**. It is not the recommended entry point. Both the observed two-step test and planned longer run use NVIDIA's official model restoration, data-loading strategy, optimizer, precision, and checkpointing implementation. The source and original YAML remain unchanged; experiment-specific choices belong in our launcher.

```text
Brev host
├── NVIDIA driver / L4
├── ~/work/nemo-speech-src      → /nemo-src (read-only, v3.0.0 recipes)
├── ~/data/nsc                 → /data/nsc (read-only data)
├── ~/hf-cache                 → /root/.cache/huggingface (read-only checkpoint)
└── ~/work/nemotron-poc/results/official_finetune/<run-id>
                               → /results (writable, persistent)

NeMo Speech container
pretrained .nemo → official fine-tuning entry point + Hydra overrides
                → Lhotse train / validation loaders → GPU optimizer steps
                → checkpoints + .nemo export + logs on host storage
```

Train remains the **300-utterance / 111-speaker / ~0.66 h** seed-42 subset; validation remains **50 utterances / 50 speakers / ~0.11 h**, with no shared speakers or utterance IDs. Host manifests are `/home/ubuntu/data/nsc/poc/nemo/train_300.jsonl` and `dev_50.jsonl`; inside the container they are `/data/nsc/poc/nemo/train_300.jsonl` and `dev_50.jsonl`. Both retain the five fields documented above, with `lang` and `target_lang` set to `en-US`. Original datasets remain unchanged.

#### Duration audit: eligibility is not consumption

The operator inspected durations in the actual 300-record training manifest:

| Check | Observed value |
| --- | ---: |
| Recordings | 300 |
| Shortest | 1.43 s |
| Median | 6.23 s |
| P95 | 20.25 s |
| Longest | 29.72 s |
| Eligible under the earlier proposed `max_duration=12.0` | 240; 60 excluded (20%) |
| Eligible under the final `max_duration=39.99` | All 300 by manifest duration |

The final cutoff matches the upstream maximum-duration default. Reducing duration would change the workload distribution and could hide memory problems involving longer audio. Here we reduce **optimizer steps**, retaining the intended training route and duration eligibility. **All 300 eligible does not mean all 300 were consumed:** the run stopped after two optimizer steps. No audit of every recording's training consumption or full-run memory demand is claimed.

#### Training settings and rationale

The [repository launcher](../../scripts/guide-02/run_nsc_train.sh) preserves the supplied command history and both successful Hydra corrections. It is **reconstructed**, not compared byte-for-byte with `/home/ubuntu/work/nemotron-poc/run_nsc_train.sh` or independently tested on an L4. The exact operator-supplied `baseline_eval.py` is now retained in Git. Adding files here does not synchronize the Brev workspace.

| Setting | Recorded choice | Why / boundary |
| --- | --- | --- |
| Entry / initial model | Official `speech_to_text_finetune.py`; cached pretrained `.nemo` | Exercise the intended longer-run implementation, restoring pretrained weights |
| GPU / precision | One L4; `bf16-mixed` | BF16 mixed precision is appropriate on supported L4 hardware; no measured memory/performance gain claimed |
| Optimizer / learning rate | AdamW; `1e-5` | Conservative initial fine-tuning choice, not an optimized learning rate |
| Scheduler | `'~model.optim.sched'` | Delete the key for constant learning rate; do not leave `sched=null` |
| Training batch size | `+model.train_ds.batch_size=1` | Conservative starting point for L4 memory |
| Gradient accumulation | Inherited NVIDIA configuration; no new override | Effective composed run value has not been independently verified; do not infer an effective batch size |
| Training / validation loaders | Upstream Lhotse enabled; `default_prompt_mode=langID` for both | Prompt-aware training using the recorded `en-US` metadata |
| Audio loading | `is_tarred=false` | Prepared audio consists of individual FLAC recordings |
| Batching | `batch_duration=null`; `quadratic_duration=null` | Replace duration-based batching with the recorded batch-size-one setup |
| Duration / workers | `max_duration=39.99`; `num_workers=0` for train/validation | Retain all selected duration values; keep data-loading concurrency simple |
| Smoke budget / validation | `max_steps=2`; `val_check_interval=2`; `limit_val_batches=2` | Two optimizer steps and only two validation batches; not evaluation of all 50 |
| Checkpointing | Experiment-manager callback; `save_top_k=1` and model export | Exercise saving to persistent host storage; restoration is a separate check |

The pinned YAML includes duration-based batching and a Noam warmup of **10,000 steps**, intended for a larger training budget. Those defaults are not assumed ideal for one L4 and 300 records. Constant `1e-5` is a deliberate initial experiment choice; choose any future schedule from the actual training budget and validation evidence. Training uses Lhotse; the successful **offline inference** workaround used `use_lhotse=False`. These are separate execution paths, not contradictory settings.

#### Launcher, mounts, and modes

Follow the [launcher prerequisites and usage](../../scripts/guide-02/README.md#official-training-launcher). The host paths must exist on the SSH-connected node; committing the script to GitHub does not deploy it there.

| Command part | Purpose |
| --- | --- |
| `--rm --gpus all` | Remove the stopped container while exposing the GPU |
| `"$SOURCE:/nemo-src:ro"` | Make the pinned official recipes visible without modifying upstream source |
| `"$DATA:/data/nsc:ro"` | Protect original and derived datasets from container writes |
| `"$CACHE:/root/.cache/huggingface:ro"` | Restore the already-downloaded pretrained checkpoint without changing cache contents |
| `"$OUTPUT:/results"` | Persist experiment output after the container exits |
| `--config-path` / `--config-name` | Select the official YAML; Hydra overrides supply POC settings |
| `+init_from_nemo_model` | Restore the specific cached pretrained export |
| `exp_manager` overrides | Name the experiment/version and save checkpoints beneath `/results` |
| `2>&1 \| tee "$OUTPUT/console.log"` | Capture console output on the host; `pipefail` retains Docker failure status |

The checkpoint path is pinned to observed Hugging Face snapshot `ea30d66debe3740a08b573244286791d423d6b3e`. It supports reproducing this run **only when that file exists in the mounted cache**; a new node may need a different checkpoint path. No portable replacement or image digest is invented.

`smoke` uses two steps, validation every two training batches, at most two validation batches, and logging each step. `full <steps>` uses the same core training route with a chosen step budget, `val_check_interval=1.0`, `limit_val_batches=1.0`, and logging every ten steps. The primary intended difference is duration, with validation/logging limits also explicitly changed; fractional `1.0` is different from integer `2`. `full 600` is an **unexecuted example**, not a result. Run IDs use UTC timestamps to the second, and outputs stay outside Git.

#### Troubleshooting history: failures and fixes in order

##### A. Installed framework without example recipes

Searches under `/opt` and `/workspace` did not locate the expected fine-tuning examples. A broader search found the installed NeMo package at `/workspace/nemo`. Package availability and example-recipe availability are separate checks.

The operator cloned the pinned source on the host:

```bash
git clone --depth 1 --branch v3.0.0 \
  https://github.com/NVIDIA-NeMo/Speech.git \
  ~/work/nemo-speech-src
```

The checkout was confirmed at `/home/ubuntu/work/nemo-speech-src` and mounted read-only. Keep the container framework, upstream recipes, experiment launcher, data, and outputs in separate locations.

##### B. Training script initially absent from the container path

Observed error:

```text
python: can't open file
'/nemo-src/examples/asr/speech_to_text_finetune.py':
[Errno 2] No such file or directory
```

Host-path and bind-mount investigation eventually produced this successful diagnostic:

```bash
docker run --rm \
  -v "$HOME/work/nemo-speech-src:/nemo-src:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  bash -lc '
    test -f /nemo-src/examples/asr/speech_to_text_finetune.py \
    && echo "SUCCESS: Training script accessible via -v mount"
  '
```

Observed output: `SUCCESS: Training script accessible via -v mount`. A diagnostic using `--mount` also found the script. The exact earlier cause was **not conclusively established**; both mount forms subsequently worked. A successful clone does not prove container visibility: check host and container paths separately.

##### C. Root-owned host results directory

Observed error and ownership:

```text
mkdir: cannot create directory
'/home/ubuntu/work/nemotron-poc/results/official_finetune':
Permission denied

drwxr-xr-x ... root root ... results
```

The operator inspected and corrected only the parent results directory:

```bash
ls -ld ~/work/nemotron-poc/results
sudo chown ubuntu:ubuntu ~/work/nemotron-poc/results
```

This was deliberately **non-recursive**. It does not establish ownership of every new output file or the normalized/NeMo manifests. Containers running as UID 0 can create root-owned bind-mount outputs. A customer deployment needs an intentional UID/GID and persistent-storage ownership policy.

##### D. Hydra override of a missing key

Observed composition error:

```text
hydra.errors.ConfigCompositionException:
Could not override 'trainer.limit_val_batches'.

To append to your config use:
+trainer.limit_val_batches=2
```

The YAML did not define that key. The initial `trainer.limit_val_batches="$VAL_BATCHES"` attempted to override it; the successful launcher uses **`+trainer.limit_val_batches="$VAL_BATCHES"`**.

| Hydra expression | Meaning |
| --- | --- |
| `key=value` | Override an existing key |
| `+key=value` | Add a missing key |
| `++key=value` | Add or override a key |
| `~key` | Delete a key |

These [Hydra override semantics](https://hydra.cc/docs/advanced/override_grammar/basic/) are part of the runtime contract: valid Bash syntax does not establish successful configuration composition.

##### E. Null scheduler retained a key NeMo tried to modify

Observed container traceback:

```text
File "/workspace/nemo/core/classes/modelPT.py",
line 691, in setup_optimization

optim_config['sched']['max_steps'] = self._trainer.max_steps

TypeError: 'NoneType' object does not support item assignment
```

`model.optim.sched=null` retained the scheduler key with a `None` value. NeMo checked key presence and then attempted to modify that value. **`'~model.optim.sched'`** deletes the key and allowed the run to proceed with the deliberately chosen constant learning rate.

The [pinned ModelPT source](https://github.com/NVIDIA-NeMo/Speech/blob/v3.0.0/nemo/core/classes/modelPT.py) contains the same key-presence/write behavior, at a different line number from the container traceback. This supports the explanation, not byte identity between the container package and host source tag. No NVIDIA framework code was patched. Future scheduling belongs in experiment configuration and should fit the longer-run budget.

#### Observed execution and logs

The operator executed:

```bash
bash ~/work/nemotron-poc/run_nsc_train.sh smoke
```

Run ID: **`smoke-2-20261009T092152Z`**. Reported excerpts:

```text
Validation DataLoader 0:
50% | 1/2

Epoch 0, global step 2:
'val_wer' reached 1.66667 (best 1.66667)

Saving model to:
'/results/nemotron_nsc/smoke-2-20261009T092152Z/checkpoints/nemotron_nsc--val_wer=1.6667-epoch=0.ckpt'

New .nemo model saved to:
/results/nemotron_nsc/smoke-2-20261009T092152Z/checkpoints/nemotron_nsc.nemo

Trainer.fit stopped: max_steps=2 reached.
```

The final message establishes completion of the two-step limit. Logs also showed repeated checkpoint/export activity during finalization. Training-step timing, restoration, validation, serialization, and total wall-clock time are different measurements. No peak VRAM, GPU utilization, throughput, convergence, end-to-end runtime, or cost benchmark was captured.

<a id="checkpoint-artifacts-saved-not-yet-restored"></a>

#### Checkpoint artifacts: saved; export subsequently restored

The operator checked persistent host files with:

```bash
find ~/work/nemotron-poc/results/official_finetune \
  -type f \( -name "*.nemo" -o -name "*.ckpt" \) \
  -printf '%p  (%s bytes)\n'
```

Host checkpoint directory:

```text
/home/ubuntu/work/nemotron-poc/results/official_finetune/smoke-2-20261009T092152Z/nemotron_nsc/smoke-2-20261009T092152Z/checkpoints/
```

The outer run directory belongs to the launcher; `nemotron_nsc/<run-id>/checkpoints` is the experiment-manager nesting inside that mount.

| Observed file | Reported size |
| --- | ---: |
| `nemotron_nsc--val_wer=1.6667-epoch=0.ckpt` | 7,660,848,351 bytes |
| `nemotron_nsc.nemo` | 2,553,098,240 bytes |
| `nemotron_nsc--val_wer=1.6667-epoch=0-last.ckpt` | 7,660,848,415 bytes |

All three files were observed on the host during the October 9 training milestone. These sizes are historical measurements, not newly measured in the October 10 evaluation session. File existence alone does not establish restoration; [stage 8](#8-checkpoint-comparison-and-development) subsequently confirmed restoration and inference from the `.nemo` export. Resuming the `.ckpt` training state remains untested.

A `.ckpt` is a Lightning/NeMo training checkpoint intended for restoring training state where the recipe supports it. A `.nemo` is a model export for restoration/inference. The exports here are approximately **7.66 GB** per training checkpoint versus **2.55 GB** for the model export (decimal units). Training checkpoints may carry optimizer and other training state as well as weights; their exact contents have not been inspected. The chosen fine-tuning entry point initializes from `.nemo`, not a demonstrated `.ckpt` resume.

Checkpointing has storage/I/O requirements independent of GPU compute. Consider persistent capacity, write throughput, checkpoint frequency, retention, recovery requirements, and whether an artifact actually restores. The model files and console logs remain on the host, outside Git.

#### Why smoke-test validation WER is not the baseline comparison

The training log's `val_wer=1.66667` is a fraction, approximately **166.67%**. It is **not directly comparable** with the pretrained **10.84%** baseline:

- Smoke validation used only **two batches**; baseline evaluation used all **50 fixed recordings**.
- Training-time WER follows a different evaluation path from the separate baseline evaluator's identical reference/prediction normalization.
- Raw output can include language tags, capitalization, and punctuation; the baseline excludes these scoring differences while preserving raw text.
- Only two optimizer steps ran to prove functionality, not recognition-quality improvement.

Do not infer improved or degraded model quality from this training-time score. The subsequent [offline comparison](#offline-comparison-and-interpretation) uses the fixed-50 evaluator. A valid comparison changes **only the checkpoint**: keep the same 50 recordings, prompt/transcription settings, decoding behavior, normalization, and scoring implementation. The existing baseline remains **50 recordings / 793 reference words / 10.84% offline WER**.

The checkpoint filename contains `val_wer=1.6667` because the current callback monitors training-time validation WER. **TODO before longer training:** confirm an appropriate checkpoint-selection metric and its normalization/decoding policy. Successful checkpoint saving is not evidence of useful model selection.

## 8. Checkpoint comparison and development

<a id="next-restore-and-evaluate-the-exported-checkpoint--planned"></a>

### Export evaluation and baseline reproduction — complete

**Operator-reported execution: October 10, 2026.** The Brev Ubuntu host's NVIDIA L4 and `nvcr.io/nvidia/nemo-speech:26.07.00` were used for both offline evaluations. The maintainer has no access to the Brev host or result files and did not independently execute GPU inference or recalculate WER. Commands and outputs below come from the supplied execution evidence.

#### Environment and evaluator verification

The operator confirmed the following checks passed, without supplying their terminal outputs. No new GPU readings, artifact sizes, or directory listings are inferred:

```bash
nvidia-smi
cd ~/work/nemotron-poc
pwd
ls -lh
find ~/work/nemotron-poc/results/official_finetune \
  -type f \( -name "*.nemo" -o -name "*.ckpt" \) \
  -printf '%p  (%s bytes)\n'
ls -lh ~/work/nemotron-poc/results/baseline/
wc -l ~/data/nsc/poc/nemo/dev_50.jsonl
```

The expected baseline files are `baseline_predictions.jsonl` and `baseline_metrics.json`; the fixed validation manifest contains 50 records per the completed evaluations.

The operator then inspected the existing **238-line** evaluator:

```bash
nl -ba baseline_eval.py | sed -n '1,260p'
```

It already supports positional `manifest`, optional `--model-path`, `--run-name`, and `--output-dir`. No Python modification or second evaluator was needed. The reported loading logic is:

```python
if args.model_path:
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=args.model_path
    )
else:
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=MODEL_NAME
    )
model.cuda()
model.eval()
```

Both evaluations use `RNNTPromptTranscribeConfig(use_lhotse=False, batch_size=1, num_workers=0, target_lang="en-US", verbose=False)` and `model.transcribe(audio=audio_paths, override_config=transcribe_cfg)` inside `torch.inference_mode()`. `from_pretrained()` loads the original NVIDIA model; `restore_from()` loads the local NeMo export. `model.cuda()` moves parameters to the GPU, `model.eval()` selects evaluation behavior, and `torch.inference_mode()` disables autograd tracking to avoid inference overhead. Neither evaluation call resumes training or updates parameters. Identical normalization prevents case, punctuation, and tag differences from masquerading as checkpoint quality changes.

Reference and prediction text use the [baseline normalization policy](#reported-evaluation-implementation-and-scoring-policy): remove language tags, lowercase, normalize curly apostrophes, remove punctuation except apostrophes, and collapse whitespace. **No number or abbreviation normalization is applied.** NeMo computes corpus-level WER with:

```python
wer = word_error_rate(
    hypotheses=normalized_preds,
    references=normalized_refs,
)
```

This aggregates word errors over reference words rather than averaging per-recording WER. Output names are `<run_name>_predictions.jsonl` and `<run_name>_metrics.json`. The complete operator-supplied source is now retained unchanged as [baseline_eval.py](../../scripts/guide-02/baseline_eval.py). The maintainer did not directly compare bytes against the inaccessible host file. The script opens output files for writing: keep separate run names/directories to preserve historical results; reusing both would overwrite those outputs.

#### Original baseline metrics inspected

The operator read:

```bash
cat ~/work/nemotron-poc/results/baseline/baseline_metrics.json
```

Supplied output:

```json
{
  "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
  "manifest": "/data/nsc/poc/nemo/dev_50.jsonl",
  "evaluation_mode": "offline_transcribe",
  "num_utterances": 50,
  "total_reference_words": 793,
  "target_lang": "en-US",
  "batch_size": 1,
  "use_lhotse": false,
  "normalization": "lowercase, strip language tags, remove punctuation except apostrophes, normalize whitespace",
  "wer": 0.10844892812105927,
  "wer_percent": 10.844892812105927
}
```

These original metrics imply **86 word-level errors / 793 reference words**. The normalization description is retained verbatim from the metrics; operator script inspection additionally confirmed curly-apostrophe handling and no number/abbreviation normalization.

#### Fine-tuned export: location and executed evaluation

The October 9 training run `smoke-2-20261009T092152Z` completed two optimizer steps. The operator located its model export:

```bash
find results/official_finetune -type f -name "nemotron_nsc.nemo"
```

Observed relative host path:

```text
results/official_finetune/smoke-2-20261009T092152Z/nemotron_nsc/smoke-2-20261009T092152Z/checkpoints/nemotron_nsc.nemo
```

Executed on the SSH-connected Brev host:

```bash
mkdir -p ~/work/nemotron-poc/results/finetuned_eval

docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  -v "$HOME/work/nemotron-poc/results/finetuned_eval:/results" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --model-path /work/results/official_finetune/smoke-2-20261009T092152Z/nemotron_nsc/smoke-2-20261009T092152Z/checkpoints/nemotron_nsc.nemo \
  --run-name finetuned_smoke2 \
  --output-dir /results
```

Aggregate excerpt of supplied terminal output (prediction text omitted):

```text
Transcribing 50 recordings...

=== EVALUATION RESULTS ===
Run name       : finetuned_smoke2
Utterances     : 50
Reference words: 793
WER            : 10.97%

=== SAVED FILES ===
/results/finetuned_smoke2_predictions.jsonl
/results/finetuned_smoke2_metrics.json

Evaluation completed.
```

This confirms successful `.nemo` restoration and inference on all 50 validation recordings. **Only rounded 10.97% WER was supplied**; the full-precision fine-tuned metrics JSON was not available and is not reconstructed.

#### Pretrained baseline: executed recheck

Executed after the fine-tuned evaluation on the same Brev host:

```bash
mkdir -p ~/work/nemotron-poc/results/baseline_recheck

docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  -v "$HOME/work/nemotron-poc/results/baseline_recheck:/results" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --run-name baseline_recheck \
  --output-dir /results
```

Omitting `--model-path` selects `ASRModel.from_pretrained(model_name="nvidia/nemotron-3.5-asr-streaming-0.6b")`. The same current evaluator, container tag, manifest, batch size, language prompt, normalization, and corpus-level scoring were used for both runs.

Aggregate excerpt of supplied output:

```text
Transcribing 50 recordings...

=== EVALUATION RESULTS ===
Run name       : baseline_recheck
Utterances     : 50
Reference words: 793
WER            : 10.84%

=== SAVED FILES ===
/results/baseline_recheck_predictions.jsonl
/results/baseline_recheck_metrics.json

Evaluation completed.
```

The original **10.84% WER was reproduced to two decimal places**. Complete prediction files and full-precision recheck metrics were not supplied or compared, so this is not a claim of bit-for-bit reproducibility.

#### Mounts and persistent results

Both commands execute Python **inside the NeMo Speech container**, not the host Python environment. `--gpus all` exposes the L4. Dataset and project mounts are read-only; the separate writable `/results` mount retains outputs after `--rm`. The Hugging Face cache persists on the host. No training occurs during these commands.

| Host location | Container location | Use |
| --- | --- | --- |
| `~/hf-cache` | `/root/.cache/huggingface` | Persistent model cache |
| `~/data/nsc` | `/data/nsc` | Read-only audio/manifests |
| `~/work/nemotron-poc` | `/work` | Read-only evaluator and exported model |
| `~/work/nemotron-poc/results/finetuned_eval` | `/results` in fine-tuned run | Writable evaluation output |
| `~/work/nemotron-poc/results/baseline_recheck` | `/results` in baseline run | Separate writable recheck output |

The fine-tuned host directory contains `finetuned_smoke2_predictions.jsonl` and `finetuned_smoke2_metrics.json`; the recheck directory contains `baseline_recheck_predictions.jsonl` and `baseline_recheck_metrics.json`. Original `results/baseline/` and `results/official_finetune/` are preserved.

Transcript-bearing predictions and example transcripts from the supplied output are not added to Git. Dataset licensing and transcript sensitivity must be reviewed before publishing them. Existing ignore rules exclude predictions, data, caches, results, and model/checkpoint files; this guide retains aggregate results and commands.

#### Offline comparison and interpretation

| Metric | Original pretrained Nemotron | Two-step fine-tuned Nemotron |
| --- | --- | --- |
| Additional optimizer steps | 0 | 2 |
| Validation recordings | 50 | 50 |
| Reference words | 793 | 793 |
| Word-level errors | 86 from original metrics | 87 inferred from normally rounded WER |
| Measured WER | 10.84% (original and recheck) | 10.97% (terminal only) |
| Evaluation mode | Offline transcription | Offline transcription |
| Container | NeMo Speech 26.07.00 | NeMo Speech 26.07.00 |
| Target language / batch size | `en-US` / 1 | `en-US` / 1 |
| Lhotse transcription | Disabled | Disabled |

If the supplied fine-tuned percentage was rounded normally, the unique integer error count consistent with 10.97% over 793 words is 87. Arithmetic from those counts gives approximately **+0.1261 percentage points**, or **1.16% relative increase** in word errors. These are derived comparisons, not additional full-precision metrics read from the fine-tuned JSON.

Under that rounding assumption, the fine-tuned export made one more word-level error on this small development set (approximately **+0.13 percentage points**). No statistical significance was assessed; this is not evidence of a statistically meaningful regression. Two optimizer steps are insufficient evidence of meaningful Singapore-English adaptation; the result does not establish that fine-tuning generally reduces quality. **The training-to-export-to-restoration-to-evaluation workflow works; this smoke test did not improve transcription accuracy.** It is not a production-ready model, held-out result, or streaming benchmark.

In supplied recording 4, the fine-tuned output matched the reference word `require` where the pretrained model produced `required`. Other substitutions remained in that utterance, and corpus WER was still slightly higher. This single-word observation is not a substitute for whole-dataset scoring. WER measures agreement with spoken reference text, not grammatical correctness; standardized English can introduce errors against Singapore-English constructions. Full example transcripts/prediction files remain outside Git.

#### Solutions Architect lessons

The completed lifecycle demonstrates containerized GPU execution, persistent artifacts, protected inputs, restoration, and consistent ASR evaluation. The [decision framework](#solutions-architect-decision-framework) connects these checks to customer requirements and trade-offs; the [lessons learned](#lessons-learned-for-customer-decisions) incorporate the subsequent readiness audit and pilot rationale.

<a id="next-review-the-launcher-before-longer-fine-tuning--planned"></a>

### Pre-training readiness review — complete

**Operator-reported October 10, 2026.** After the fixed-50 comparison, the operator audited the local NVIDIA configuration and training manifest before committing more GPU time. The configuration excerpt and the supplied pilot case are available; the complete current Brev launcher, source checkout, and data remain inaccessible to the maintainer. Repository inspection can compare the supplied case with the repo, but cannot establish full host-file equality.

#### Official configuration audit

Executed on the Brev Ubuntu host:

```bash
CFG="$HOME/work/nemo-speech-src/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml"

grep -nE '^[[:space:]]*(accumulate_grad_batches|limit_train_batches|check_val_every_n_epoch|monitor|mode|always_save_nemo|save_top_k|max_epochs|max_steps|val_check_interval|min_duration|max_duration|batch_size|use_lhotse):' "$CFG"
```

Supplied output from the locally installed NVIDIA YAML:

```text
149:    use_lhotse: true
156:    max_duration: 39.99
157:    min_duration: 0.1
186:    batch_size: 2
192:    use_lhotse: true
202:    batch_size: 16
207:    use_lhotse: true
351:  max_epochs: -1
352:  max_steps: 500000
353:  val_check_interval: 0.5
358:  accumulate_grad_batches: 1
369:  limit_train_batches: 1000
378:    monitor: "val_wer"
379:    mode: "min"
380:    save_top_k: 5
381:    always_save_nemo: True
```

These are **source defaults**, not the resolved pilot configuration. The existing launcher overrides batch size, maximum steps, validation interval/coverage, and retention. Training uses Lhotse; the separate offline evaluator disables it for transcription.

| Setting | Audited source / launcher relationship | Implication |
| --- | --- | --- |
| Gradient accumulation | Source `accumulate_grad_batches=1`; launcher batch size 1 and one GPU | Expected effective batch is one example per optimizer step, subject to actual dataloader behavior and resolved config |
| Validation interval | Source `0.5`; existing `full` overrides to float `1.0`; pilot overrides to integer `50` | Short experiments need validation before an epoch ends |
| Validation coverage | Smoke `2` batches; pilot `1.0` | Pilot requests the full validation dataloader; verify actual records/batches in logs |
| Epoch/step limits | Source `max_epochs=-1`, `max_steps=500000`; pilot overrides steps to `100` | Source's large step limit is not an executed experiment |
| Training batch cap | Source `limit_train_batches=1000` | Maximum batches per epoch; does not force an epoch to contain 1,000 batches |
| Checkpoint selection | Source `monitor="val_wer"`, `mode="min"`; launcher `save_top_k=1` | Intends to retain the lowest internal validation WER checkpoint |
| Export | Source `always_save_nemo=True` | Verify which weights the exported .nemo contains; export is not proven to be the best checkpoint |

A floating validation interval `1.0` schedules checks at epoch end; an integer `50` selects a training-batch interval. Similarly, validation limit `1.0` requests full coverage, while integer `2` limits coverage to two batches. Accumulation counts batches before optimizer updates. See [Lightning Trainer settings](https://lightning.ai/docs/pytorch/stable/common/trainer) and [gradient accumulation](https://lightning.ai/docs/pytorch/stable/common/optimization). Validate the actual events in the pinned runtime; intervals in batches are not inherently optimizer-step intervals.

The source audit explains why the existing full mode could provide no intermediate validation before a short step-limited run ends. The pilot targeted validation near steps 50 and 100 with accumulation one. The supplied excerpt confirms a step-100 metric/checkpoint; step 50, actual coverage, and dataloader behavior remain full-log investigation tasks. The audit does not establish completed checkpoint selection or export provenance. Internal `val_wer` still differs from the fixed-50 offline score; compare restored artifacts with the established evaluator.

#### Training-manifest duration assessment

The following **metadata-only Python check ran on the Brev host**. It does not load NeMo or run training/inference; all model execution remains inside the pinned container.

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = Path.home() / "data/nsc/poc/nemo/train_300.jsonl"

with manifest.open() as f:
    samples = [json.loads(line) for line in f if line.strip()]

eligible = [
    s for s in samples
    if 0.1 <= float(s["duration"]) <= 39.99
]

print("Total recordings:", len(samples))
print("Eligible recordings:", len(eligible))
print("Excluded by duration:", len(samples) - len(eligible))
print(
    "Total eligible audio hours:",
    round(sum(float(s["duration"]) for s in eligible) / 3600, 3)
)
print("Shortest recording:", min(float(s["duration"]) for s in samples))
print("Longest recording:", max(float(s["duration"]) for s in samples))
PY
```

Supplied output:

```text
Total recordings: 300
Eligible recordings: 300
Excluded by duration: 0
Total eligible audio hours: 0.656
Shortest recording: 1.43
Longest recording: 29.72
```

All 300 records fall within `0.1–39.99` seconds; none are excluded solely by these duration boundaries. The rounded total is **0.656 hours**, approximately **39.4 minutes** and **7.87 seconds per recording**. The latter figures are arithmetic estimates from the rounded total, not independently measured audio properties. This confirms metadata eligibility, not runtime consumption of every example, audio integrity, or sufficient adaptation data.

The earlier subset check established **111 training speakers** and 50 separate validation speakers. Breadth alone does not establish representative speaker, vocabulary, or recording-condition coverage. Around 39 minutes is a small adaptation dataset for this approximately 600-million-parameter model. Overfitting, limited vocabulary coverage, and weak generalization remain risks to assess through training/validation behavior and representative held-out testing. Keep the fixed-50 set for development, and retain NSC held-out plus GigaSpeech evaluation after model selection.

<a id="controlled-100-step-pilot--prepared-execution-unconfirmed"></a>

### Controlled 100-step pilot — complete

**Operator-reported execution: October 10, 2026.** Run `pilot-100-20261010T100537Z` completed on the Brev NVIDIA L4 using the pinned NeMo Speech container, official training script, original pretrained Nemotron weights, and the existing Lhotse training route. The maintainer did not execute training/inference on Brev or independently inspect the artifacts. The existing [launcher](../../scripts/guide-02/run_nsc_train.sh) already matches the supplied case; no new pilot implementation or training-argument change is needed.

The separate mode preserves smoke/full and their historical settings. It was chosen to obtain early optimization, validation, and export evidence before committing to 300 or 600 steps. Its configured values were:

| Pilot setting | Value / evidence boundary |
| --- | --- |
| Initial model | Original pretrained `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Model class | `EncDecRNNTBPEModelWithPrompt` |
| Optimizer / LR | AdamW / `1e-5`; existing scheduler-key deletion |
| GPU / precision | One L4, approximately 24 GB / `bf16-mixed` |
| Training / validation manifests | 300 / 50 recordings |
| Training batch size / accumulation | Launcher setting 1 / inherited 1 |
| Optimizer steps | 100, completion confirmed by trainer stop message |
| Validation interval / coverage | Integer 50 training batches / `1.0` full validation loader requested |
| Logging | Every 10 steps |
| Checkpoint policy | Internal `val_wer`, minimum, `save_top_k=1` |
| Run output | `~/work/nemotron-poc/results/official_finetune/pilot-100-20261010T100537Z/` |

#### Executed launch and observed training output

Operator-reported host sequence:

```bash
cd ~/work/nemotron-poc
bash -n run_nsc_train.sh
bash run_nsc_train.sh pilot
```

The launcher expands the documented `pilot` case (`STEPS=100`, `VAL_INTERVAL=50`, `VAL_BATCHES=1.0`, `LOG_INTERVAL=10`) into the shared Docker/Hydra command. All model execution occurred inside `nvcr.io/nvidia/nemo-speech:26.07.00`; host Python was used only for metadata/results analysis.

Key supplied output:

```text
Epoch 0: 10% | 100/1000

Epoch 0, global step 100:
'val_wer' reached 0.12863 (best 0.12863)

Trainer.fit stopped: max_steps=100 reached.
```

The run completed normally at 100 optimizer steps; no fatal exception was reported. Internal `val_wer=0.12863` is approximately **12.863%**, measured through the training validation path. It is not directly comparable with standardized offline WER. The excerpt explicitly establishes validation/checkpoint reporting at **global step 100**; it does **not** establish a step-50 validation event or its score.

Supplied checkpoint/export messages included:

```text
Epoch 0, global step 100:
'val_wer' reached 0.12863 (best 0.12863),
saving model to
'/results/nemotron_nsc/pilot-100-20261010T100537Z/checkpoints/nemotron_nsc--val_wer=0.1286-epoch=0.ckpt'
as top 1

New .nemo model saved to:
 /results/nemotron_nsc/pilot-100-20261010T100537Z/checkpoints/nemotron_nsc.nemo
```

The export was updated multiple times during final checkpoint handling; the last supplied messages showed export activity around **10:16 UTC (18:16 Singapore time)**. Those timestamps and progress estimates are not a complete performance report. No peak VRAM, utilization, throughput, cost, or end-to-end elapsed runtime is established.

#### Saved pilot artifacts

Executed on the Brev host:

```bash
MODEL="$HOME/work/nemotron-poc/results/official_finetune/pilot-100-20261010T100537Z/nemotron_nsc/pilot-100-20261010T100537Z/checkpoints/nemotron_nsc.nemo"

ls -lh "$MODEL"
ls -lh "$(dirname "$MODEL")"
```

Supplied rounded listing excerpt:

```text
total 17G
7.2G  nemotron_nsc--val_wer=0.1286-epoch=0-last.ckpt
7.2G  nemotron_nsc--val_wer=0.1286-epoch=0.ckpt
2.4G  nemotron_nsc.nemo
```

These are human-readable `ls -lh` values, **not exact byte counts or decimal-GB measurements**. The directory reports approximately `17G` for the observed artifacts. Persistent capacity, retention, and experiment cleanup matter independently of GPU compute; the listing does not measure storage throughput.

The monitored `.ckpt` was retained as top 1 according to internal validation; the `-last.ckpt` is the final training checkpoint. The `.nemo` export is the model restored for inference below. Its exact relationship to the best-versus-last checkpoint weights remains unverified, as does training-state recovery from either `.ckpt`. Keep these artifacts, console logs, datasets, and cache outside Git.

#### Standardized pilot export evaluation

Executed through the pinned container from the Brev host:

```bash
docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  -v "$HOME/work/nemotron-poc/results/pilot_100_eval:/results" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_eval.py \
  /data/nsc/poc/nemo/dev_50.jsonl \
  --model-path /work/results/official_finetune/pilot-100-20261010T100537Z/nemotron_nsc/pilot-100-20261010T100537Z/checkpoints/nemotron_nsc.nemo \
  --run-name pilot_100 \
  --output-dir /results
```

Observed output:

```text
Transcribing 50 recordings...

=== EVALUATION RESULTS ===
Run name       : pilot_100
Utterances     : 50
Reference words: 793
WER            : 10.47%

=== SAVED FILES ===
/results/pilot_100_predictions.jsonl
/results/pilot_100_metrics.json

Evaluation completed.
```

The model export was successfully restored and evaluated on all 50 fixed development recordings with the same evaluator, `en-US`, batch size one, disabled Lhotse transcription, normalization, and corpus WER as the baseline. Dataset and project are read-only; writable `/results` maps to host `~/work/nemotron-poc/results/pilot_100_eval/`, preserving both named files after container exit. The full pilot metrics JSON was not supplied; **10.47% is the displayed offline result**. The recording-level analysis below independently reported the word-error count within the operator's analysis, not within a maintainer rerun.

| Model | Additional optimizer steps | Recordings / reference words | Word errors | Offline WER |
| --- | ---: | --- | ---: | ---: |
| Original pretrained Nemotron | 0 | 50 / 793 | 86 | 10.84% |
| Two-step smoke export | 2 | 50 / 793 | 87 inferred from normally rounded WER | 10.97% |
| 100-step pilot export | 100 | 50 / 793 | 83 from operator recording analysis | 10.47% |

The pilot reduced error count by **3**, equivalent to approximately **0.38 percentage points** (`3 / 793 × 100`) and **3.49% relative reduction** (`3 / 86 × 100`) versus pretrained. This is preliminary improvement on a small, development-influenced set. No statistical significance, Singapore-English generalization, held-out quality, production suitability, or streaming-performance improvement is established.

### Recording-level error comparison — complete

The operator performed a **read-only, GPU-free host Python analysis** of:

```text
~/work/nemotron-poc/results/baseline_recheck/baseline_recheck_predictions.jsonl
~/work/nemotron-poc/results/pilot_100_eval/pilot_100_predictions.jsonl
```

It loaded both JSONL files, verified the same 50 recordings and matching normalized references, computed word-level edit distance per recording, compared errors/changed predictions, and aggregated totals. The exact analysis source and bulk outputs were not supplied in this handoff; the procedure and observed summary are recorded without reconstructing an allegedly executed script.

```text
=== SUMMARY ===
Improved recordings: 9
Worsened recordings: 7
Unchanged error counts: 34

Baseline errors: 86
Pilot errors: 83

Net errors reduced: 3
```

**Unchanged means equal error counts**, not identical prediction text. Some predictions changed without changing the number of errors. The analysis confirms 83 errors over 793 reference words, which rounds to 10.47%; full-precision metric-file fields are not invented.

Selected operator observations, without full transcripts:

| Recording | Baseline → pilot errors | Qualitative observation |
| --- | --- | --- |
| 4 | 3 → 2 | Matched reference `require` instead of `required` |
| 6 | 2 → 0 | Recognized a customer transaction phrase involving added desserts |
| 13 | 1 → 0 | Improved a conversational phrase |
| 14 | 1 → 0 | Improved filler recognition |
| 34 | 1 → 0 | Recognized `mm` |
| 37 | 1 → 0 | Restored a missing word |
| 47 | 5 → 3 | Improved repeated speech |
| 12 | 2 → 3 | Introduced another omission |
| 27 | 3 → 4 | Changed `inquire` to `enquire`, scored as a substitution |
| 28, 32, 44 | 1 → 2; 2 → 3; 2 → 3 | Recognition worsened |
| 49 | 3 → 5 | More errors in a statement involving leave duration |

This table is selected evidence, not an exhaustive accounting of all 9 improved and 7 worsened recordings. Improvements include conversational content, repeated words, and hesitation markers; some regressions concern numbers or potentially business-critical information. Three net errors hide these mixed changes.

For a customer-service POC, supplement aggregate WER with errors involving amounts, product names, customer requests, dates/durations, and entitlement or policy information. Correct transaction content may matter more than filler fidelity. The customer must define those priorities and acceptance criteria; the current aggregate score does not establish business utility.

The fixed scoring policy treats `inquire` and `enquire` as different words; that substitution may reflect spelling conventions rather than a recognition failure. **Do not retroactively change normalization for this comparison.** Any future policy change requires rescoring all compared models consistently. Complete transcripts, prediction JSONL, and raw audio remain outside Git.

### Next: investigate training behavior and validation logs — planned

**Latest stopping point:** pilot training/export/evaluation and recording-level comparison are complete. Before any 300- or 600-step experiment, investigate validation scheduling, Lhotse sampling/exposure, training loss, and checkpoint/export provenance. No such longer experiment is documented as completed.

The progress indicator showed **epoch 0, 100/1000**, despite 300 training records. The source audit found `limit_train_batches=1000`. A repeating Lhotse stream with a trainer-defined pseudo-epoch is a **possible explanation**, not a confirmed cause for this run. Determine whether data repeated, whether all 300 recordings were sampled, whether 100 steps represented 100 distinct examples, and whether the trainer cap defined the displayed epoch length. Do not describe this as one-third of an ordinary epoch.

The configured interval was 50 training batches with full validation requested. Step-100 validation/checkpoint reporting is confirmed; step 50 and actual validation coverage need full-log inspection. The following **prepared read-only host command has not been reported executed**:

```bash
python3 - <<'PY'
from pathlib import Path

log = (
    Path.home()
    / "work/nemotron-poc/results/official_finetune"
    / "pilot-100-20261010T100537Z/console.log"
)
text = log.read_text(errors="replace").replace("\r", "\n")
keywords = [
    "val_wer",
    "global step",
    "checkpoint save",
    "new .nemo",
    "trainer.fit",
    "validation",
]
matches = []
for line in text.splitlines():
    if any(word in line.lower() for word in keywords):
        matches.append(line.strip())

print("=== TRAINING EVENTS ===")
for line in matches[-40:]:
    print(line[:300])
PY
```

This command is a first event scan; its last-40 filter is not a complete coverage/provenance audit. Questions still requiring evidence:

1. Did validation run at both intended intervals, and what were the metrics/coverage?
2. Did training loss decrease, and how were training examples sampled/repeated?
3. Were best/last checkpoint weights different, and which weights were exported?
4. How was elapsed time divided among training, validation, and checkpoint saving?
5. What were actual resource requirements and costs? No measured peak VRAM, utilization, throughput, or cost report is supplied.

**Recommended customer decision:** continue the POC, but review this evidence before increasing step count. Then choose among more training, broader/better data, configuration changes, or a more representative development evaluation. Preserve current artifacts and frozen comparison policies. Reserve held-out NSC and GigaSpeech for the finalized model/configuration; production quality, streaming latency, concurrency, throughput, memory under target load, and deployment cost remain unvalidated.

### Lessons learned for customer decisions

1. **Training success and accuracy improvement are separate.** Two steps proved feasibility; 10.97% versus 10.84% did not demonstrate better recognition.
2. **Reproduce evaluation before interpreting a small change.** The shared evaluator/container reproduced baseline WER at the reported precision, increasing confidence in the comparison without proving exact output equality.
3. **Small development sets are sensitive to individual errors.** Smoke added one error; pilot removed three over 793 words. The pilot improved 9 recordings and worsened 7; no statistically meaningful or generalizable gain is established.
4. **Match validation frequency to experiment duration.** The pilot uses a batch interval to seek intermediate evidence; inspect actual events rather than assume epoch-end validation will occur.
5. **Internal and offline WER are different measurements.** Smoke's two-batch internal score is not interchangeable with the full fixed-50 normalized score.
6. **Measure data volume and coverage.** 300 records represent about 39 minutes; speaker breadth, terminology, and recording conditions need assessment alongside counts.
7. **Containers support a repeatable execution environment.** Read-only inputs and persistent writable results separate software, data, and artifacts; retain tags/source versions and capture runtime configuration.
8. **Verify exports and selected weights.** Smoke and pilot exports restored successfully; their exact relationship to best-versus-last checkpoint weights remains unverified.
9. **Each GPU experiment should answer a decision question.** Pilot completion, exports, and early offline improvement are established. Validation history, data exposure, training loss, resources, and provenance still need investigation before larger expenditure.
10. **Production recommendations require workload evidence.** L4 feasibility does not prove optimal sizing; customer acceptance requires representative quality, latency, concurrency, cost, and operational evidence.

## 9. Held-out NSC test

**Planned; not executed.** This stage follows the validation/development loop in stage 8.

### Freeze the model and evaluation configuration

Choose the checkpoint using validation, then record the language prompt, decoding settings, reference/prediction normalization, and WER implementation. Freeze those choices before using test results. Smoke and pilot exports have been restored and compared; the pilot shows preliminary fixed-development-set improvement. Any 300-/600-step experiment, final checkpoint selection, and configuration freeze remain pending.

### Prepare and evaluate the held-out NSC test

`nsc_test` is the identified official NSC Part 6 test source. Local download, preparation, overlap verification, and evaluation remain pending. Prepare it on the SSH-connected compute node and verify speaker/utterance-ID separation from the development data.

Evaluate the finalized system using the frozen policy, retaining raw predictions, normalized predictions, and metadata outside Git. Test results describe held-out in-domain performance; repeated tuning on them would turn this set into another validation set. See the [data roles](#data-strategy-and-experiment-overview) for that boundary.

## 10. External benchmark

**Planned; not executed.** Run this after stage 9's internal test evaluation.

`gigaspeech_test` is the identified separate-corpus benchmark for out-of-domain generalization and regressions. Local download/preparation and evaluation remain pending. Prepare and inspect it on the compute node, record its reference/scoring policy, and compare pretrained/final checkpoints consistently.

Use the frozen model/configuration rather than routinely tuning on benchmark results. Keep raw/normalized predictions and experiment metadata outside Git. This check does not establish universal or production performance.

True streaming, latency/performance measurements, workload packaging, and SLURM submission remain later milestones; they are not completed by the offline test/benchmark sequence.

## Completion boundaries and next step

The ten numbered stages give the execution order; this table records the individual checks within them. Supporting diagnostics and later streaming work are listed separately from completed agenda stages.

| Individual milestone | Status |
| --- | --- |
| Node resource preflight | Complete |
| NeMo container validation | Complete: NeMo 3.0.0, ASR import and model loading |
| Model download and loading | Complete: `EncDecRNNTBPEModelWithPrompt` |
| Persistent Hugging Face cache | Host directory and bind mount configured; files survive container removal |
| Model placement on GPU | Complete: `cuda:0` |
| Optional generic WAV appendix | Instructions prepared; this alternative sample remains unexecuted |
| NSC training/query download and extraction | Complete: 2,289 records, 214M |
| NSC dev download and extraction | Complete: 1,316 records |
| Four-stage data strategy and workflow | Documented; full train/validation/test/benchmark experiment pending |
| Source train/dev manifest inspection | Complete for reported source checks; full-source audio integrity not audited |
| Speaker-overlap verification | Complete for train/validation: 111 / 64 speakers, 0 overlap; NSC test checks pending |
| Utterance-ID overlap verification | Complete for train/validation: 0 overlap; NSC test checks pending |
| Annotation-token inventory and counts | Complete for both full source manifests: `<v-noise>`, `<unk>`, `<noise>` |
| `<unk>` eligibility impact analysis | Complete: 2,158 / 1,225 records would remain, 4.59 / 2.71 h |
| POC `<unk>`/annotation-handling decision | Complete: exclude whole `<unk>` utterances; keep noise-tagged utterances for later token removal |
| Deterministic speaker-aware subset generation | Complete: seed 42; generator reported 300 train / 50 validation |
| Derived POC directory creation | Complete: `~/data/nsc/poc` |
| Derived POC manifest creation | Complete: `train_300.jsonl` / `dev_50.jsonl` |
| Generated-file ownership correction | Complete: only the two derived files changed from `root:root` to `ubuntu:ubuntu` |
| Independent host-side line counts | Complete: 300 train / 50 validation |
| Derived annotation-tag validation | Complete: `<unk>` absent; train `<v-noise>` 87 / `<noise>` 5, validation `<v-noise>` 7 |
| Derived speaker/utterance-ID validation | Complete: 111 / 50 speakers; 0 shared speakers and 0 shared IDs |
| Independent derived-subset validation | Complete for the ownership, count, tag, speaker, and separation checks above |
| Transcript normalization output | Complete: 300/50 records; 61/4 texts changed; approved noise tags removed; independent rerun failed on missing helper |
| Annotation-policy application | Complete for selected data: whole `<unk>` utterances excluded; noise-token cleanup applied; baseline scoring policy recorded |
| Nemotron manifest conversion | Complete: five fields, `en-US`; 300/50 records and selected file existence reported |
| Output-manifest sample inspection | Complete: first record from each NeMo manifest inspected |
| New normalized/NeMo output ownership | Not reported; earlier ownership correction covered only original POC subset files |
| Single-utterance pretrained NSC GPU inference | Complete and verified on October 9, 2026: `cuda:0`; reference `uh correct` → raw prediction `Uh correct. <en-US>` |
| Fixed 50-utterance baseline / validation WER | Complete on October 9, 2026: 50 recordings, 793 reference words, 10.84% offline WER |
| Operator-supplied baseline evaluator source in Git | Complete: supplied 238-line source retained unchanged; local syntax/stubbed checks, no independent GPU run |
| Official training smoke test | Complete on October 9, 2026: two optimizer steps on L4, limited validation executed |
| Further/full fine-tuning | Pending; two-step smoke and 100-step pilot complete; no 300-/600-step experiment completed |
| Checkpoint/model saving | Complete: two .ckpt files and one .nemo export observed with byte sizes |
| Export restoration and inference | Complete: operator restored the selected .nemo and transcribed all 50 recordings; .ckpt resume remains untested |
| Fine-tuned fixed 50-record WER | Complete on October 10: 793 reference words / 10.97% offline WER |
| Pretrained baseline reproduction | Complete on October 10: same evaluator/settings, 10.84% WER; complete predictions not compared for equality |
| Training configuration audit | Complete from operator YAML output and supplied launcher case; full current host file not byte-compared |
| Training duration eligibility assessment | Complete: 300 total / 300 eligible / 0 excluded / 0.656 h by manifest metadata |
| 100-step pilot configuration | Complete: integer interval 50, full internal validation requested, logging every 10 steps |
| 100-step pilot execution | Complete October 10: pilot-100-20261010T100537Z; max_steps=100 reached |
| Pilot artifacts/export | Complete: monitored and last .ckpt, .nemo; rounded listing total 17G; export weight provenance pending |
| Pilot restoration/offline evaluation | Complete: 50 recordings / 793 words / 10.47% WER |
| Recording-level comparison | Complete: 86 → 83 errors; 9 improved / 7 worsened / 34 equal-error-count recordings |
| Training behavior/validation investigation | Next; step-50 event, sampling/exposure, loss, and export provenance unresolved |
| Checkpoint-selection policy audit | Complete: internal val_wer/min, launcher save_top_k=1; actual best-checkpoint/export provenance verification pending |
| Checkpoint/model comparison on validation | Complete: pretrained 10.84%, smoke 10.97%, pilot 10.47%; final model selection pending |
| Post-training evaluation | Complete for smoke and pilot .nemo exports; future-run evaluation pending |
| Final model/configuration | Not yet finalized |
| NSC held-out test (`nsc_test`) | Not started; no local download/evaluation documented; prepare/verify and evaluate after configuration freeze |
| GigaSpeech external/OOD benchmark (`gigaspeech_test`) | Not started; no local download/evaluation documented; prepare/verify and evaluate after NSC test |
| Performance benchmarking | Not yet complete |
| True streaming inference | Not yet complete |
| Streaming latency benchmarking | Not yet complete |
| Manual annotation/audio review | Future work; excluded from this POC |
| NeMo Curator curation | Future work; no installation or run performed |

### End-of-day checkpoint

**Current checkpoint (October 10, 2026):** The end-to-end two-step smoke-test workflow is validated: training → checkpoint/export saving → .nemo restoration → offline inference/evaluation. The export from `smoke-2-20261009T092152Z` scored **10.97% WER**; the same evaluator reproduced the pretrained **10.84% WER** on **50 recordings / 793 reference words**. The pipeline works, but this run did not improve transcription accuracy. The operator-supplied evaluator is now retained unchanged in Git. The subsequent configuration audit and duration eligibility assessment are complete: **300/300 records, 0.656 h**. The **100-step pilot completed**, exported/restored successfully, and scored **10.47% offline WER**. Read-only recording analysis reported **86 → 83 errors**, with **9 improved / 7 worsened / 34 unchanged error counts**.

**Next session — stage 8:** Inspect the pilot console log and training behavior to establish step-50 validation, actual coverage/sampling, loss trajectory, best/last/export weight identity, and resource/time evidence. A read-only event-scan command is prepared above but unexecuted. Preserve artifacts; do not jump to 300 or 600 steps before reviewing this evidence.

Later: longer fine-tuning → validation/development loop → freeze checkpoint/configuration/scoring → `nsc_test` → `gigaspeech_test` → streaming latency/performance. These remain pending. The infrastructure progression remains Docker validation → actual Nemotron inference → streaming inference → package the workload → SLURM; streaming and scheduler work remain future milestones.

## Troubleshooting notes

### Troubleshooting: arbitrary container UID

An initial attempt added the host UID/GID option to make generated files inherit the `ubuntu` account's ownership:

```bash
--user "$(id -u):$(id -g)"
```

It first failed with:

```text
exec: python3: not found
```

The interpreter was located at `/opt/venv/bin/python3`, but invoking it under the arbitrary host UID then failed with:

```text
/opt/venv/bin/python3: Permission denied
```

Do not assume a vendor container supports arbitrary host UID/GID execution. This POC uses the image's default context with read-only source/tooling and a writable derived-output mount. Ownership was corrected only for the original two subset files; no check/correction was reported for the newer normalized or NeMo files.

### Missing runtime helper

After normalization, an independent tag-inspection rerun failed:

```text
python3: can't open file '/work/inspect_transcript_tags.py': [Errno 2] No such file or directory
```

The operator checked the host workspace:

```bash
ls -lh ~/work/nemotron-poc/*.py
```

At that point it contained `check_poc_eligibility.py`, `check_split_overlap.py`, `create_poc_subsets.py`, and `normalize_transcripts.py`; the inspection helper was absent. Presence in GitHub and presence in the remote runtime are separate facts. The audit failed, and no successful suggested `grep` result was reported. This does not replace the earlier subset inventory or normalization reports with a new independent audit.

### Unexpected output: inspect the actual script

During node validation, `check_split_overlap.py` was accidentally overwritten with transcript-tag inspection code. Unexpected output prompted inspection of the script, preservation of the incorrect copy as `check_split_overlap.py.bad` locally, restoration of the known-good version, and rerunning validation. The incorrect artifact is not committed. Inspect the program producing unexpected output before drawing conclusions about the data.

### NeMo Curator — future work

The operator considered Curator after reviewing the [NVIDIA article](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-for-saudi-arabic-dialects-with-a-path-to-other-languages/), which used audio standardization and quality scoring/filtering. We chose to keep this small POC's completed preparation pipeline; no Curator installation or curation run was performed. A future experiment can inspect quality-score distributions while preserving challenging but usable Singapore English and keeping evaluation sets fixed. Do not transfer Arabic-corpus thresholds directly to NSC. Manual listening/relabeling of possible `<unk>` issues also remains future work.

<a id="single-file-inference-reference--not-yet-executed"></a>

## Appendix: optional single-file inference smoke test — not yet executed

These previously prepared commands remain an optional generic English one-file alternative, not evidence of completed inference. The [NSC validation smoke test](#single-utterance-pretrained-gpu-inference--complete) above is now verified; the [50-utterance offline baseline/WER](#pretrained-50-utterance-validation-baseline--complete) is also complete. This older generic example does not include the explicit non-Lhotse workaround and may encounter the reported prompt error in the pinned container; follow the successful configuration above when adapting it. Run these commands in the GPU host's shell over SSH; no notebook, GUI, or microphone is required. This is whole-file inference with a streaming-capable model; true streaming is a later milestone.

### 1. Create an audio directory and download a small sample

```bash
mkdir -p ~/asr-audio

curl -fL \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav \
  -o ~/asr-audio/sample.wav
```

If `curl` is unavailable, use:

```bash
wget -O ~/asr-audio/sample.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

This file is linked in an [NVIDIA NeMo ASR example](https://github.com/NVIDIA-NeMo/Speech/discussions/3553). The guide maintainer verified its download and format: mono, 16 kHz, approximately 7.4 seconds, 238 kB. It is a general English smoke-test sample, not evidence of Singapore English adaptation. `curl -fL` fails on HTTP errors and follows redirects; `-o` chooses the local filename. Audio stays outside the repository.

### 2. Load the model on GPU and print the transcript

```bash
docker run --rm --gpus all -i \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/asr-audio:/audio:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python - <<'PYTHON'
import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b"
)
model.cuda()
model.eval()
print("Model device:", next(model.parameters()).device)

with torch.inference_mode():
    results = model.transcribe(
        audio=["/audio/sample.wav"],
        batch_size=1,
        return_hypotheses=True,
        target_lang="en-US",
    )

print("Transcript:", results[0].text)
PYTHON
```

The [NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) documents NeMo loading and transcription. The [prompt-conditioned NeMo model source](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/rnnt_bpe_models_prompt.py) supports the `audio`, `return_hypotheses`, and language-prompt arguments used here.

| Command part | Purpose |
| --- | --- |
| `docker run` | Starts a container on the GPU host |
| `--rm` | Removes the stopped container; host-mounted cache and audio remain |
| `--gpus all` | Makes the host GPUs available to the container |
| `-i` | Keeps standard input open so the Python code can enter the container |
| Cache `-v` mount | Reuses the persistent host Hugging Face cache |
| Audio `-v` mount | Maps `~/asr-audio` on the host to `/audio` in the container; `:ro` makes it read-only |
| `nvcr.io/nvidia/nemo-speech:26.07.00` | Reuses the NeMo Speech image already validated for model loading |
| `python -` | Runs Python code read from standard input |
| `<<'PYTHON' ... PYTHON` | Sends the multiline code without host-shell variable expansion; no interactive terminal (`-t`) is needed |

`model.cuda()` moves the model to GPU 0; `model.eval()` selects evaluation behavior, and `torch.inference_mode()` disables gradient tracking. `batch_size=1` keeps this to one file. `target_lang="en-US"` supplies the English prompt for this sample, without claiming a Singapore-specific adaptation. `return_hypotheses=True` returns a hypothesis whose `.text` is printed.

Success means the process exits normally, reports `Model device: cuda:0`, and prints a nonempty transcript after `Transcript:`. Record the actual text and any errors after running it. No transcript or inference success is claimed for this optional generic WAV example. If the file is missing, check the host download and `/audio` mount first; if model loading succeeds but transcription fails, retain the error for targeted diagnosis.
