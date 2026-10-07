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

**Goal:** Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on NSC Part 6 using the NVIDIA L4 Brev instance. This small POC follows **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**.

1. [Verify source manifests on the node](docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) — counts checked; referenced-audio integrity not yet reported.
2. [Verify speaker and utterance-ID separation](docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) — complete for train/validation; NSC test checks pending.
3. [Inspect and count transcript annotation tokens](docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-audit--complete) and [measure POC eligibility](docs/guides/02-nemotron-streaming-singapore-english.md#poc-eligibility-impact--complete) — complete; annotation policy decided.
4. [Create deterministic speaker-aware POC subsets](docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) — 300/50 generated; independent validation pending.
5. Finalize remaining normalization rules and apply the chosen annotation policy reproducibly to derived data.
6. Convert derived data to NeMo manifests with container-visible audio paths.
7. Run baseline inference on validation and record WER.
8. Fine-tune on train only, starting with a training smoke test.
9. Use validation for development/model selection.
10. Freeze the model/configuration, normalization, and decoding settings.
11. Evaluate `nsc_test` for final held-out Singapore-English results.
12. Evaluate `gigaspeech_test` for external/OOD generalization and regressions after the NSC test.

**Status:** Preflight, NeMo/model loading, source inspection/separation, annotation auditing, eligibility analysis, and the POC policy are complete. Speaker-aware generation reported **300 train utterances / 111 speakers / 0.66 h** and **50 validation utterances / 50 speakers / 0.11 h**, using seed 42. The derived files exist; independent validation, normalized output, and steps 6–12 remain pending. NSC test/GigaSpeech preparation and evaluation are unvalidated.

**Current next step:** [Validate the derived subsets](docs/guides/02-nemotron-streaming-singapore-english.md#next-validate-derived-subsets--pending): ownership, independent line counts, `<unk>` absence, speaker counts, and deterministic reproduction. Then finish normalization and convert NeMo manifests. Preserve originals; manual relabeling is excluded from this POC. The optional one-WAV test remains unexecuted.

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
│       ├── check_split_overlap.py
│       ├── inspect_transcript_tags.py
│       ├── check_poc_eligibility.py
│       └── create_poc_subsets.py
└── docs/
    └── guides/
        ├── README.md         # Short directory index
        ├── 01-brev-gpu-node-validation.md
        └── 02-nemotron-streaming-singapore-english.md
```

The [written guides](docs/guides/README.md) contain commands, architecture explanations, and recorded evidence. Guide 02's [overlap check](scripts/guide-02/check_split_overlap.py), [annotation audit](scripts/guide-02/inspect_transcript_tags.py), [eligibility check](scripts/guide-02/check_poc_eligibility.py), and [subset generator](scripts/guide-02/create_poc_subsets.py) are reusable helpers used in recorded node runs. The [progress log](PROGRESS.md) keeps historical milestones; the [roadmap](ROADMAP.md) holds future topics. Add scripts or configuration when an actual guide step needs them, and link from that guide.

## Documentation standard

Distinguish observed results from prepared instructions and planned work. Record the environment, reproduction steps, troubleshooting lessons, and costs/cleanup where known. Publish reviewed, sanitized evidence; keep credentials, datasets, model caches/weights, raw recordings, and raw workload outputs outside Git. A local laptop path is not automatically available on an SSH-connected GPU host.

This is an independent learning guide and portfolio.
