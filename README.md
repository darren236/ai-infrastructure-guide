# AI Infrastructure Guide

Practical AI infrastructure guides and a technical portfolio demonstrating NVIDIA Solutions Architect skills: GPU stack validation, containerized workloads, troubleshooting, and clear explanations of architecture and operational tradeoffs.

Start with **Guide 01** to understand and validate the GPU stack, then follow **Guide 02** into a small Nemotron ASR fine-tuning POC. Each guide's agenda is below; open the guide for commands, explanations, observed results, and pending work.

Use the [current progress](PROGRESS.md#current-state) for the latest checkpoint, [milestone history](PROGRESS.md#milestone-history) for earlier evidence, [helper-script index](scripts/guide-02/README.md) for runnable tooling, and [roadmap](ROADMAP.md) for future topics.

## Guides and agendas

<a id="agenda-from-physical-gpu-to-application"></a>
<a id="guides"></a>

### Guide 01 — GPU Node Validation

[Open Guide 01](docs/guides/01-brev-gpu-node-validation.md)

**Goal:** Understand the path from the provisioned GPU to a working application on an NVIDIA L4 Brev node. Follow these eight layers as the reading agenda; use the minimum relevant check during customer troubleshooting.

1. [Physical GPU](docs/guides/01-brev-gpu-node-validation.md#1-physical-gpu) — identify the provisioned GPU.
2. [PCIe enumeration](docs/guides/01-brev-gpu-node-validation.md#2-pcie-enumeration) — check device visibility with `lspci`.
3. [Linux kernel](docs/guides/01-brev-gpu-node-validation.md#3-linux-kernel) — identify kernel, OS, and architecture.
4. [NVIDIA driver](docs/guides/01-brev-gpu-node-validation.md#4-nvidia-driver) — use `nvidia-smi`; its CUDA field reports driver compatibility, not the installed Toolkit/runtime version.
5. [CUDA runtime / Toolkit](docs/guides/01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit) — inspect `nvcc` in a devel container; a host Toolkit is not required for this setup.
6. [Container runtime](docs/guides/01-brev-gpu-node-validation.md#6-container-runtime) — validate `docker info` and GPU-enabled `docker run`.
7. [PyTorch / NeMo](docs/guides/01-brev-gpu-node-validation.md#7-pytorch--nemo) — check framework CUDA access and model placement.
8. [Application](docs/guides/01-brev-gpu-node-validation.md#8-application) — process real inputs and verify the output; continue into Guide 02.

**Status:** Host/driver visibility, container GPU access, compiler inspection, and framework access are validated for the reported checks. Guide 02 verifies pretrained NSC GPU inference, a 50-record offline baseline, and an official two-step training smoke test with saved artifacts. Standalone CUDA kernel/tensor checks, longer training, and benchmarks remain pending; Guide 01 is in progress.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

**Goal:** Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on NSC Part 6 using the NVIDIA L4 Brev instance. This small POC follows **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**.

Follow these ten stages from node readiness to final evaluation. Each link gives the commands and evidence; training/validation can repeat during development.

1. [Node resources](docs/guides/02-nemotron-streaming-singapore-english.md#node-preflight--complete) — check disk, host RAM, and GPU availability.
2. [NeMo container](docs/guides/02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete) — check the pinned Speech container's NeMo version and imports.
3. [Pretrained model](docs/guides/02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete) — restore Nemotron with persistent caching and confirm GPU placement.
4. [Data strategy and source checks](docs/guides/02-nemotron-streaming-singapore-english.md#data-strategy-and-experiment-overview) — define data roles and verify records, split separation, and annotation tags.
5. [Training and validation inputs](docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) — build deterministic subsets, normalize text, and verify NeMo audio paths and language fields.
6. [Pretrained inference and baseline](docs/guides/02-nemotron-streaming-singapore-english.md#pretrained-50-utterance-validation-baseline--complete) — verify one transcription, then establish offline WER on fixed validation.
7. [Training smoke test and artifacts](docs/guides/02-nemotron-streaming-singapore-english.md#official-nemo-fine-tuning--two-step-gpu-smoke-test) — run the official GPU smoke test and verify persistent checkpoint files.
8. [Checkpoint comparison and development](docs/guides/02-nemotron-streaming-singapore-english.md#next-restore-and-evaluate-the-exported-checkpoint--planned) — restore and compare on fixed validation; iterate longer training and model selection.
9. [Held-out NSC test](docs/guides/02-nemotron-streaming-singapore-english.md#evaluation-preparation--todo) — freeze model, decoding, and scoring choices before evaluating `nsc_test`.
10. [External benchmark](docs/guides/02-nemotron-streaming-singapore-english.md#evaluation-preparation--todo) — check `gigaspeech_test` for generalization/regressions after the NSC test, without routine tuning.

**Current position:** Setup, train/validation preparation, pretrained inference/baseline, and two-step training with artifact saving are verified for the reported scope. Baseline: **50 recordings / 10.84% offline WER**. Longer training, checkpoint comparison, held-out evaluation, and streaming/performance remain pending.

**Next check — stage 8:** Restore/evaluate the saved export on the same 50 recordings under identical baseline policies. The exact evaluator source is still awaited; saved artifacts have not yet been reloaded or shown to improve quality. See [current progress](PROGRESS.md#guide-02-end-of-day-checkpoint) and [completion boundaries](docs/guides/02-nemotron-streaming-singapore-english.md#completion-boundaries-and-next-step).

## Future guides

Storage/data access → Slurm → distributed training/NCCL → Kubernetes/GPU Operator → observability → architecture synthesis. These topics are planned in the [roadmap](ROADMAP.md); guide documents are added when work begins.

## Repository layout

```text
ai-infrastructure-guide/
├── README.md                 # Guide overview and agendas
├── ROADMAP.md                # Future topics
├── PROGRESS.md               # Current status and dated milestone history
├── .gitignore
├── scripts/
│   └── guide-02/
│       ├── README.md         # Helper purposes, inputs, and node setup
│       ├── check_split_overlap.py
│       ├── inspect_transcript_tags.py
│       ├── check_poc_eligibility.py
│       ├── create_poc_subsets.py
│       ├── normalize_transcripts.py
│       ├── convert_to_nemo_manifest.py
│       ├── baseline_smoke.py
│       └── run_nsc_train.sh
└── docs/
    └── guides/
        ├── README.md         # Short directory index
        ├── 01-brev-gpu-node-validation.md
        └── 02-nemotron-streaming-singapore-english.md
```

The [written guides](docs/guides/README.md) contain commands, architecture explanations, and recorded evidence. Runnable assets live under `scripts/guide-NN/`; the [Guide 02 tooling index](scripts/guide-02/README.md) explains six preparation helpers, the one-record GPU inference script, and the official-training launcher. The inference script implements the reported workflow; the training launcher is reconstructed from supplied command history. Neither has been compared byte-for-byte with the Brev copy or independently GPU-tested here. Repository scripts and the Brev workspace are separate locations; a commit does not deploy a helper to the node. The [progress log](PROGRESS.md) separates current status from historical milestones; the [roadmap](ROADMAP.md) holds future topics.

## Documentation standard

Distinguish observed results from prepared instructions and planned work. Record the environment, reproduction steps, troubleshooting lessons, and costs/cleanup where known. Publish reviewed, sanitized evidence; keep credentials, datasets, model caches/weights, raw recordings, and raw workload outputs outside Git. A local laptop path is not automatically available on an SSH-connected GPU host.

This is an independent learning guide and portfolio.
