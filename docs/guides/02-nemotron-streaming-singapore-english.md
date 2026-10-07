# Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[All guides](README.md) · [Repository overview](../../README.md)

## Where this guide fits in the agenda

[Guide 01](01-brev-gpu-node-validation.md#agenda-validate-the-stack-from-gpu-to-application) follows physical GPU → PCIe enumeration → Linux kernel → NVIDIA driver → CUDA runtime/Toolkit → container runtime → PyTorch/NeMo → application. The NeMo import and model-placement checks below complete the reported framework access checks at **layer 7**. This guide continues into **layer 8: application** with a small ASR POC; successful transcription, training, and evaluation are still pending.

## Goal and scope

Build a small proof of concept (POC) and tutorial for fine-tuning `nvidia/nemotron-3.5-asr-streaming-0.6b` on Singapore National Speech Corpus (NSC) Part 6, using an NVIDIA L4 24 GB class Brev instance. The goal is a manageable learning exercise, not production-quality tuning. Model loading, dataset preparation, training, and evaluation results are documented only after they are performed.

## Data strategy and experiment overview

Follow four data roles: **train → validation → test → external benchmark**. Only the NSC query and dev sources have been downloaded and extracted. The approximately 300/50 POC subsets have **not** been created. In this guide, **validation** and **dev** mean the same role; the source directory remains `nsc_dev_3h`.

| Stage | Current source | POC size | Purpose | Model learns from it? | When used | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| Train | `nsc_query_5h` | ~300 utterances | Fine-tuning | Yes, directly through gradient updates | First learning stage | Source downloaded; subset not created |
| Validation | `nsc_dev_3h` | ~50 utterances | Baseline comparison, tuning, checkpoint/model decisions | No gradients; influences development indirectly | Before and during fine-tuning development | Source downloaded; subset not created |
| Test | TBD | TBD | Final held-out internal evaluation | No gradients or development tuning | After validation and development decisions are finalized | Not selected |
| External benchmark | TBD | TBD | Independent generalization check | No gradients or routine tuning | After internal test evaluation | Not selected; optional later milestone |

```text
Training data
     │
     ↓
Fine-tune model: learn weights through gradient updates
     │
     ↓
Validation data
     ├── compare baseline vs fine-tuned model
     ├── tune development choices
     ├── choose checkpoint
     └── finalize model/configuration
     │
     ↓
Test data
     └── final held-out internal evaluation
     │
     ↓
External benchmark
     └── independent generalization check
```

Capture baseline **word error rate (WER)** on validation before the first fine-tuning run. Return to the same validation subset to compare checkpoints and development choices. Training and validation can repeat during development; final test evaluation comes after those decisions are settled.

### Available sources: origin and contents

Both sources are NSC Part 6 extracts distributed through [pengyizhou/IALP-2026-data](https://huggingface.co/datasets/pengyizhou/IALP-2026-data). The intended experiment roles below are our POC choices; the source name `query` does not prevent us from using that subset for fine-tuning.

| Source | Full source size | Contents and intended use |
| --- | --- | --- |
| `nsc_query_5h` | 2,289 utterances; ~5 hours; ~214 MB, recorded as `214M` by `du -sh` | `.flac` audio and `manifest.jsonl`; source for ~300 training utterances |
| `nsc_dev_3h` | 1,316 utterances; ~3 hours | `.flac` audio and `manifest.jsonl`; source for ~50 validation utterances; transcripts may contain `<v-noise>` |

Each JSONL line describes an utterance: `id` identifies it, `speaker` identifies the speaker, `duration` gives seconds, `text` is the reference transcript, and `audio` is a relative path into `audio/`. The download commands and example records are below. Hour totals are approximate source descriptions, not independently summed durations. The small subset sizes are planned counts, not measured subset hours or proof of production readiness.

### Direct learning versus development decisions

Training data changes model weights through backpropagation. Validation data must never enter gradient updates, but its results still influence the final system indirectly: an engineer may choose a learning rate, checkpoint, normalization rule, or decoding setting because it improves validation WER. Finalize those choices using validation, and record them before testing.

Test results are for evaluating the finalized internal result. If we repeatedly tune choices based on test results, that set effectively becomes another validation set and loses its role as an independent final check. A new untouched holdout would be needed for a fresh final evaluation. See the [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) for this distinction.

### What results can tell us

- **Train:** Lower training loss shows better fit to training examples; it does not establish performance on unseen speech.
- **Validation:** WER on the fixed ~50 utterances supports baseline comparisons and development decisions under recorded scoring rules. It is a small, development-influenced result, not an unbiased final estimate for all Singapore English.
- **Test:** A properly separated, untouched set evaluates the finalized system on held-out internal data. Conclusions apply to that set's size and distribution; they do not establish production performance or generalization to every customer.
- **External benchmark:** A separate source, corpus, or real-world target distribution checks whether improvements extend beyond the data used during development. Results support conclusions about that benchmark, not universal generalization. Keep it independent of the internal train/validation/test workflow and out of routine tuning.

### Later data decisions — TODO

- **Final test:** No dedicated test set is selected or prepared. After validation choices are finalized, select and verify a suitable holdout with audio and reference transcripts, separated from POC train/validation speakers and utterance IDs. Record its source and size, then evaluate the fixed model/configuration without tuning on its results. If reserving data earlier, keep it untouched during development.
- **External benchmark:** No benchmark is selected or prepared. Later, optionally choose independent audio and reference transcripts from a different source, corpus, or target distribution, document its scope, and evaluate after the internal test. Its source, size, and scoring rules remain TBD.

## Guide 02 workflow agenda

Node preflight and model loading are recorded below. Follow this application workflow next; none of the future execution steps is complete.

1. Understand the four data roles: train, validation, test, external benchmark.
2. Inspect the available NSC data — sources downloaded and example records inspected; full checks pending.
3. Create deterministic POC train/validation subsets — approximately 300/50 utterances, not created.
4. Verify speaker and utterance-ID separation before inference or training; revisit when selecting a test set.
5. Define transcript normalization, including treatment of annotation tags; freeze scoring rules for comparisons.
6. Convert to NeMo manifests with audio paths valid inside the container.
7. Run baseline inference on validation and record WER.
8. Fine-tune on train only, starting with a training smoke test on the L4.
9. Evaluate checkpoints/model choices on validation; record any development changes and compare under the same scoring rules.
10. Finalize the model/configuration: checkpoint, normalization, and decoding settings.
11. Select/verify and use a held-out test set for final internal evaluation — TBD.
12. Optionally evaluate an external benchmark after the internal test — TBD.

Perform data preparation in the SSH-connected GPU host's working directory, outside Git. Download there and bind-mount host data into the container; a laptop path is not automatically available on the remote node. Record the source revision/checksum, fixed selection rule or seed, selected IDs, overlap-check results, normalization rules, container/model versions, and run settings. Keep validation IDs fixed; when scoring rules change during development, rescore both models consistently. These records remain to be produced.

This POC keeps the experiment small for learning. A customer deployment would choose data coverage and evaluation sizes around actual users, audio conditions, and acceptance criteria; ~300/50 utterances are not a production recommendation. WER checks recognition quality, while streaming behavior, latency, throughput, and scheduler integration remain separate future work.

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

The inspected train/validation records share the fields `id`, `speaker`, `duration`, `text`, and `audio`, with relative FLAC paths. These source manifests have not yet been converted to NeMo format. Validation transcripts can contain annotation tags such as `<v-noise>`; tag removal is planned as part of transcript normalization, whose rules have not yet been defined or applied.

### Why keep development data separate?

Evaluate development choices on utterances and speakers not used for gradient updates. Reserve the supplied dev split for validation, baseline/fine-tuned WER comparison, and checkpoint selection; use the query split for fine-tuning. The [dataset publisher](https://huggingface.co/datasets/pengyizhou/IALP-2026-data#split-construction-nsc) describes the NSC splits as speaker-disjoint. Local speaker and utterance-ID overlap checks remain pending; the two example records alone do not verify that property. Validation influences development choices, so it does not replace the later final test.

## Next: create tiny deterministic POC subsets — planned

The next execution step is to select roughly **300 training utterances** from the query split and **50 validation utterances** from the separate dev split, using a fixed selection rule or seed so the experiment can be reproduced. Record selected IDs, then verify speaker and utterance-ID separation before using the subsets. This step has not been executed.

Tiny subset creation, overlap verification, transcript normalization, NeMo manifest conversion, baseline inference, the training smoke test, fine-tuning, checkpointing, and evaluation all remain incomplete. Final test and external benchmark selection remain TODOs. Follow the workflow agenda above; the aim remains a small POC/tutorial, not production optimization.

## Single-file inference reference — not yet executed

These previously prepared commands remain available for a one-file check; they are not evidence of completed baseline inference. The current next milestone is the tiny subset selection above. Run these commands in the GPU host's shell, including when connected over SSH. They require no notebook, GUI, or microphone. This is whole-file inference with a streaming-capable model; true streaming is a later milestone.

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
| Tiny deterministic train/validation subsets | Not yet complete: approximately 300 / 50 planned |
| Speaker-overlap verification | Not yet complete |
| Utterance-ID overlap verification | Not yet complete |
| Transcript normalization | Rules not yet defined or applied |
| NeMo manifest conversion | Not yet complete |
| Annotation-tag removal | Not yet complete |
| Baseline inference / validation WER | Not yet complete |
| Training smoke test | Not yet complete |
| Fine-tuning | Not yet complete |
| Checkpointing | Not yet complete |
| Checkpoint/model comparison on validation | Not yet complete |
| Final model/configuration | Not yet finalized |
| Final held-out internal test | Not selected; evaluation pending after development choices are finalized |
| External benchmark | Not selected; optional later milestone after internal test |
| Performance benchmarking | Not yet complete |
| True streaming inference | Not yet complete |

Next, create the tiny deterministic train/validation subsets, then verify separation, define normalization, and convert manifests before baseline inference on validation. The four data roles organize later fine-tuning, model selection, internal test, and optional external evaluation; no final test or benchmark is selected. The eventual infrastructure progression remains simple Docker validation → actual Nemotron inference → streaming inference → package the workload cleanly → submit the equivalent workload through SLURM. Streaming and scheduler work remain future milestones.
