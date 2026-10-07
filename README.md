# AI Infrastructure Guide

Practical AI infrastructure guides and a technical portfolio demonstrating NVIDIA Solutions Architect skills: GPU stack validation, containerized workloads, troubleshooting, and clear explanations of architecture and operational tradeoffs.

Start with **Guide 01** to understand and validate the GPU stack, then follow **Guide 02** into a small Nemotron ASR fine-tuning POC. Each guide's agenda is below; open the guide for commands, explanations, observed results, and pending work.

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

**Status:** Host/driver visibility, container GPU access, compiler inspection, and framework access are validated for the reported checks. Actual GPU computation and application inference remain pending; Guide 01 is in progress.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

**Goal:** Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on NSC Part 6 using the NVIDIA L4 Brev instance. This is a small POC/tutorial. Its data strategy is **train → validation → final internal test → optional external benchmark**.

1. [Understand the four data roles](docs/guides/02-nemotron-streaming-singapore-english.md#data-strategy-and-experiment-overview) — train, validation, test, external benchmark.
2. [Inspect the available NSC data](docs/guides/02-nemotron-streaming-singapore-english.md#nsc-trainingquery-and-development-data--downloaded-and-extracted).
3. [Create deterministic POC train/validation subsets](docs/guides/02-nemotron-streaming-singapore-english.md#next-create-tiny-deterministic-poc-subsets--planned) — approximately 300/50 utterances.
4. Verify speaker and utterance-ID separation.
5. Define transcript normalization, including annotation-tag handling.
6. Convert to NeMo manifests with container-visible audio paths.
7. Run baseline inference on validation and record WER.
8. Fine-tune on train only, starting with a training smoke test.
9. Evaluate checkpoints/model choices on validation.
10. Finalize the model/configuration, normalization, and decoding settings.
11. Select/verify and use a held-out test set after development decisions are settled.
12. Optionally evaluate an independent external benchmark after the internal test.

**Status:** Node preflight, NeMo 3.0.0/model loading on `cuda:0`, cache configuration, and NSC query/dev downloads are complete. Only example source records have been inspected; full data checks and steps 3–12 remain pending. Test and external benchmark sources are not selected.

**Current next step:** Create the approximately **300 training / 50 validation (dev) utterance** subsets. The optional one-WAV smoke test is documented in Guide 02's appendix and remains unexecuted.

## Future guides

Storage/data access → Slurm → distributed training/NCCL → Kubernetes/GPU Operator → observability → architecture synthesis. These topics are planned in the [roadmap](ROADMAP.md); guide documents are added when work begins.

## Repository layout

```text
ai-infrastructure-guide/
├── README.md                 # Guide overview and agendas
├── ROADMAP.md                # Future topics
├── PROGRESS.md               # Current status and dated milestone history
├── .gitignore
└── docs/
    └── guides/
        ├── README.md         # Short directory index
        ├── 01-brev-gpu-node-validation.md
        └── 02-nemotron-streaming-singapore-english.md
```

The [written guides](docs/guides/README.md) contain commands, architecture explanations, and recorded evidence. The [progress log](PROGRESS.md) keeps historical milestones; the [roadmap](ROADMAP.md) holds future topics. Add reusable scripts or configuration when an actual guide step needs them, and link from that guide.

## Documentation standard

Distinguish observed results from prepared instructions and planned work. Record the environment, reproduction steps, troubleshooting lessons, and costs/cleanup where known. Publish reviewed, sanitized evidence; keep credentials, datasets, model caches/weights, raw recordings, and raw workload outputs outside Git. A local laptop path is not automatically available on an SSH-connected GPU host.

This is an independent learning guide and portfolio.
