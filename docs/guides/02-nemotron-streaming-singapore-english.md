# Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md)

## Goal and scope

Build a small proof of concept (POC) and tutorial for fine-tuning `nvidia/nemotron-3.5-asr-streaming-0.6b` on Singapore National Speech Corpus (NSC) Part 6, using an NVIDIA L4 24 GB class Brev instance. The goal is a manageable learning exercise, not production-quality tuning. Model loading, dataset preparation, training, and evaluation results are documented only after they are performed.

## Where this guide fits in the agenda

[Guide 01](01-brev-gpu-node-validation.md#agenda-validate-the-stack-from-gpu-to-application) covers host, container, and framework access. This guide continues at **layer 8: application** with the Nemotron ASR POC. NeMo import and model placement extend the reported layer 7 checks; transcription, training, and evaluation remain pending.

## Guide 02 workflow agenda

[Node preflight](#node-preflight--complete) and [NeMo/model loading](#nemo-container-and-model-on-gpu--complete) are recorded setup checks. Query/dev downloads and example inspection are complete. The [four data roles](#data-strategy-and-experiment-overview) explain the experiment; all hands-on steps below remain pending.

1. [Verify source manifests on the node](#nsc-trainingquery-and-development-data--downloaded-and-extracted): schema, record counts, and referenced audio paths.
2. [Verify speaker and utterance-ID separation](#split-construction-and-hands-on-verification); include NSC test when preparing it.
3. [Inspect and count transcript annotation tokens](#transcript-annotations-and-normalization--planned) in train and validation.
4. [Create deterministic POC train/validation subsets](#next-data-preparation-on-the-compute-node--planned) — approximately 300/50 utterances, not created.
5. Define an explicit transcript-normalization policy from the inspection; apply it reproducibly to derived data.
6. Convert derived data to NeMo manifests with audio paths valid inside the container.
7. Run baseline inference on validation and record WER.
8. Fine-tune on train only, starting with a training smoke test on the L4.
9. Use validation for development/model selection; record changes and compare checkpoints under the same scoring rules.
10. Freeze the model/configuration: checkpoint, normalization, and decoding settings.
11. Evaluate `nsc_test` for final held-out Singapore-English results — planned.
12. Evaluate `gigaspeech_test` for external/OOD generalization and regressions after the NSC test — planned.

Perform data preparation in the SSH-connected GPU host's working directory, outside Git. Download there and bind-mount host data into the container; a laptop path is not automatically available on the remote node. Record the source revision/checksum, fixed selection rule or seed, selected IDs, overlap-check results, normalization rules, container/model versions, and run settings. Keep validation IDs fixed; when scoring rules change during development, rescore both models consistently. These records remain to be produced.

This POC keeps the experiment small for learning. A customer deployment would choose data coverage and evaluation sizes around actual users, audio conditions, and acceptance criteria; ~300/50 utterances are not a production recommendation. WER checks recognition quality, while streaming behavior, latency, throughput, and scheduler integration remain separate future work.

## Data strategy and experiment overview

The intended progression is **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**. Only NSC query/dev downloads and extraction are reported; the approximately 300/50 POC subsets have **not** been created. Test and benchmark sources are now identified in the plan, with local preparation and evaluation still pending. **Validation** and **dev** mean the same role; the source directory remains `nsc_dev_3h`.

| Stage | Planned source | POC / evaluation size | Purpose | Model learns from it? | When used | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| Train | `nsc_query_5h` | ~300 utterances | Fine-tuning | Yes, directly through gradient updates | First learning stage | Source downloaded; subset not created |
| Validation | `nsc_dev_3h` | ~50 utterances | Baseline comparison, tuning, checkpoint/model decisions | No gradients; influences development indirectly | Before and during fine-tuning development | Source downloaded; subset not created |
| Test | `nsc_test` | 3,684 utterances / ~7 h, upstream | Final held-out Singapore-English evaluation | No gradients or development tuning | After model/configuration is frozen | Planned; local preparation/evaluation pending |
| External benchmark | `gigaspeech_test` | 19,930 utterances / 35.4 h, upstream | Out-of-domain (OOD) generalization/regression check | No gradients or routine tuning | After NSC test evaluation | Planned; local preparation/evaluation pending |

```text
Train: nsc_query_5h → ~300-utterance POC subset
     ↓
Gradient updates
     ↓
Validation: nsc_dev_3h → ~50-utterance POC subset
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

### Available sources: origin and contents

The [upstream dataset documentation](https://huggingface.co/datasets/pengyizhou/IALP-2026-data) identifies the four sources below. Query/dev counts were recorded locally; test/benchmark sizes are upstream descriptions, not locally verified counts or completed evaluations.

| Source | Origin | Full source size | Contents and intended use |
| --- | --- | --- | --- |
| `nsc_query_5h` | NSC IMDA Part 6 train partition | 2,289 utterances locally; ~5 h; ~214 MB, recorded as `214M` | FLAC audio and JSONL manifest; ~300 planned training utterances for gradient updates |
| `nsc_dev_3h` | NSC IMDA Part 6 train partition | 1,316 utterances locally; ~3 h | FLAC audio and JSONL manifest; ~50 planned validation utterances for development/checkpoint decisions |
| `nsc_test` | Official NSC IMDA Part 6 test partition | 3,684 utterances / ~7 h, upstream | FLAC audio and JSONL manifest, per upstream; planned held-out in-domain evaluation |
| `gigaspeech_test` | Separate GigaSpeech test corpus | 19,930 utterances / 35.4 h, upstream | WAV PCM_16 audio and JSONL manifest with normalized references, per upstream; planned external/OOD evaluation |

The downloaded query/dev JSONL records use `id`, `speaker`, `duration` (seconds), `text` (reference transcript), and `audio` (relative path into `audio/`). Their download commands and examples are below. Hour totals are source descriptions, not independently summed local durations; POC subset counts remain planned.

### Split construction and hands-on verification

`nsc_query_5h` and `nsc_dev_3h` are separate selections from the NSC Part 6 train partition, constructed using different speaker sets. **`nsc_dev_3h` is not a subset of `nsc_query_5h`**. The [publisher's split-construction notes](https://huggingface.co/datasets/pengyizhou/IALP-2026-data#split-construction-nsc) state that query, dev, and `nsc_test` are mutually speaker-disjoint, and query/dev exclude every speaker appearing in the official NSC test partition. `nsc_test` comes from that official test partition.

Keep the hands-on check: on the compute node, compare `speaker` and utterance `id` sets pairwise across query/dev/test, then verify the chosen POC subsets. Record intersections and source versions as an SA/reproducibility exercise. Upstream speaker-disjointness is a documented claim; our local speaker and utterance-ID checks remain pending.

### Direct learning versus development decisions

Training data changes model weights through backpropagation. Validation data must never enter gradient updates, but its results still influence the final system indirectly: an engineer may choose a learning rate, checkpoint, normalization rule, or decoding setting because it improves validation WER. Finalize those choices using validation, and record them before testing.

Test results are for evaluating the finalized internal result. If we repeatedly tune choices based on test results, that set effectively becomes another validation set and loses its role as an independent final check. A new untouched holdout would be needed for a fresh final evaluation. See the [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) for this distinction.

### What results can tell us

- **Train:** Lower training loss shows better fit to training examples; it does not establish performance on unseen speech.
- **Validation:** WER on the fixed ~50 utterances supports baseline comparisons and development decisions under recorded scoring rules. It is a small, development-influenced result, not an unbiased final estimate for all Singapore English.
- **Test:** `nsc_test` evaluates the frozen system on held-out NSC in-domain speech. Conclusions apply to this test distribution; they do not establish production performance or generalization to every customer.
- **External benchmark:** `gigaspeech_test` checks generalization and regressions on a separate corpus. Compare baseline and final-model results under fixed scoring rules, without routine tuning on this benchmark. Its results do not establish universal generalization.

### Evaluation preparation — TODO

- **NSC test:** Prepare `nsc_test` on the GPU host, verify local counts and overlap, and keep it out of the development loop. Evaluate only after the model, normalization, and decoding configuration are frozen.
- **GigaSpeech OOD benchmark:** Prepare `gigaspeech_test` on the host and document local counts and scoring/reference normalization. Run the planned generalization/regression comparison after NSC test evaluation. Preparation and both evaluations remain unvalidated.

## Node preflight — complete

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

### What the checks establish

- `df -h` checks filesystem capacity and usage; `-h` formats sizes in human-readable units. Check the filesystem where downloads and container storage will live. See the [GNU df manual](https://www.gnu.org/software/coreutils/manual/html_node/df-invocation.html).
- `free -h` checks CPU/system RAM and swap, separately from GPU VRAM. Linux uses otherwise unused RAM for filesystem cache, so low `free` memory can coexist with high `available` memory. `available` estimates memory usable by new applications without swapping, accounting for reclaimable cache. See the [free manual](https://www.man7.org/linux/man-pages/man1/free.1.html).
- `nvidia-smi` checks GPU/driver communication and reports GPU memory, utilization, thermals, power, and active processes. No temperature or power reading was supplied for this milestone. The reported CUDA compatibility level does not establish an installed host Toolkit version. See [NVIDIA's nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Preflight checks help rule out simple causes such as disk exhaustion, current host-memory pressure, or an already-occupied GPU before investigating model/container issues. This snapshot showed available disk and RAM and an idle GPU; it does not yet establish resource sufficiency for the selected model, data, or training configuration.

## NeMo container and model on GPU — complete

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator validated the official NeMo Speech container on the Linux GPU host:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo; print(nemo.__version__)"
```

The reported NeMo version was `3.0.0`. The operator then successfully restored the model with `ASRModel.from_pretrained(...)` and configured the persistent cache:

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

## NSC training/query and development data — downloaded and extracted

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator downloaded the small NSC Part 6 training/query dataset and the separate development split from [IALP-2026-data on Hugging Face](https://huggingface.co/datasets/pengyizhou/IALP-2026-data). The counts and records below are operator-reported observations. Keep downloaded archives and audio on the GPU host, outside the repository.

For reproduction, first choose a host working directory outside the Git checkout. Both archives extract into the current directory. For example:

```bash
mkdir -p ~/asr-data
cd ~/asr-data
```

### Training/query split

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

### Separate development split

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

The inspected train/validation records share the fields `id`, `speaker`, `duration`, `text`, and `audio`, with relative FLAC paths. These source manifests have not yet been converted to NeMo format.

### Transcript annotations and normalization — planned

**Observed:** The operator reports annotation tokens such as `<v-noise>` in **both training/query and validation/dev transcripts**. `<v-noise>` represents a vocal/non-lexical noise annotation, rather than an ordinary spoken word. Do not assume it is the only annotation type present; a full inventory and counts are still pending.

Before converting manifests or calculating WER, we will:

1. Inspect train and validation transcripts.
2. Enumerate and count annotation tokens matching `<...>` in both sources.
3. Define an explicit transcript-normalization policy based on all observed types.
4. Apply that policy consistently and reproducibly, with recorded rules for WER comparisons.
5. Preserve the original manifests unchanged; write normalized subsets and NeMo manifests to separate derived files.

These preparation steps have not been executed. No final removal or replacement rule is selected yet; inspect the annotation types before deciding how to handle them.

### Why keep development data separate?

Use query for gradient updates and dev for validation WER, development choices, and checkpoint selection. They share the NSC train partition but use different speaker sets; neither is a subset of the other. Verify separation locally as described above. Validation influences development, so it does not replace the final `nsc_test` evaluation.

<a id="next-create-tiny-deterministic-poc-subsets--planned"></a>

## Next: data preparation on the compute node — planned

First verify the source manifests, speaker/utterance-ID separation, and transcript annotation inventory on the node. Then select roughly **300 training utterances** from query and **50 validation utterances** from dev using a fixed selection rule or seed; record selected IDs and recheck subset separation. Define the normalization policy from the inspection, apply it to derived data, and convert that data to NeMo manifests before baseline validation inference.

Follow the [12-step workflow agenda](#guide-02-workflow-agenda); all of these execution steps remain pending. NSC test and GigaSpeech sources are identified in the plan, with local preparation and evaluation still pending. The aim remains a small POC/tutorial, not production optimization.

## Completion boundaries and next step

| Guide 02 stage | Status |
| --- | --- |
| Node resource preflight | Complete |
| NeMo container validation | Complete: NeMo 3.0.0, ASR import and model loading |
| Model download and loading | Complete: `EncDecRNNTBPEModelWithPrompt` |
| Persistent Hugging Face cache | Host directory and bind mount configured; files survive container removal |
| Model placement on GPU | Complete: `cuda:0` |
| Single-file WAV inference | Instructions prepared; GPU execution and transcript pending |
| NSC training/query download and extraction | Complete: 2,289 records, 214M |
| NSC dev download and extraction | Complete: 1,316 records |
| Four-stage data strategy and workflow | Documented; execution remains pending |
| Full source-manifest verification | Not yet complete |
| Speaker-overlap verification | Not yet complete |
| Utterance-ID overlap verification | Not yet complete |
| Annotation-token inventory and counts | Not yet complete; occurrence reported in both train and validation |
| Tiny deterministic train/validation subsets | Not yet complete: approximately 300 / 50 planned |
| Transcript normalization | Rules not yet defined or applied |
| Annotation-policy application | Not yet complete; final rules undecided |
| NeMo manifest conversion | Not yet complete |
| Baseline inference / validation WER | Not yet complete |
| Training smoke test | Not yet complete |
| Fine-tuning | Not yet complete |
| Checkpointing | Not yet complete |
| Checkpoint/model comparison on validation | Not yet complete |
| Final model/configuration | Not yet finalized |
| NSC held-out test (`nsc_test`) | Planned; local preparation/verification and evaluation pending after configuration freeze |
| GigaSpeech external/OOD benchmark (`gigaspeech_test`) | Planned; local preparation/verification and evaluation pending after NSC test |
| Performance benchmarking | Not yet complete |
| True streaming inference | Not yet complete |

Next, verify source manifests, separation, and annotation tokens before creating the tiny subsets, defining normalization, and converting derived manifests. Later: baseline validation → train → validation/development loop → freeze model/configuration → `nsc_test` → `gigaspeech_test`. Both evaluations remain planned. The infrastructure progression remains Docker validation → actual Nemotron inference → streaming inference → package the workload → SLURM; streaming and scheduler work remain future milestones.

<a id="single-file-inference-reference--not-yet-executed"></a>

## Appendix: optional single-file inference smoke test — not yet executed

These previously prepared commands remain available for a one-file check; they are not evidence of completed baseline inference. The current next milestone is the data-preparation sequence above. Run these commands in the GPU host's shell, including when connected over SSH. They require no notebook, GUI, or microphone. This is whole-file inference with a streaming-capable model; true streaming is a later milestone.

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

Success means the process exits normally, reports `Model device: cuda:0`, and prints a nonempty transcript after `Transcript:`. Record the actual text and any errors after running it. No transcript or inference success is claimed yet. If the file is missing, check the host download and `/audio` mount first; if model loading succeeds but transcription fails, retain the error for targeted diagnosis.
