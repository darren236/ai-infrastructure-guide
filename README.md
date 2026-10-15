# AI Infrastructure Guide

Practical AI infrastructure guides and a technical portfolio demonstrating NVIDIA Solutions Architect skills: GPU stack validation, containerized workloads, troubleshooting, and clear explanations of architecture and operational tradeoffs.

Start with **Guide 01** to understand and validate the GPU stack, follow **Guide 02** into a small Nemotron ASR fine-tuning POC, and use **Guide 04** to learn workload scheduling on the same single-node GPU environment. Each guide's agenda is below; open the guide for commands, explanations, observed results, and pending work.

Use the [current progress](PROGRESS.md#current-state) for the latest checkpoint, [milestone history](PROGRESS.md#milestone-history) for earlier evidence, [helper-script index](scripts/guide-02/README.md) for repository tooling and host-only evaluation, and [roadmap](ROADMAP.md) for future topics.

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

**Status:** Host/driver visibility, container GPU access, compiler inspection, and framework access are validated for the reported checks. Guide 02 verifies pretrained NSC GPU inference, a 50-record offline baseline, official training/export restoration, and a 3,670-record held-out comparison. Standalone CUDA kernel/tensor checks, further training beyond the completed 100-step pilot, external/streaming benchmarks, and performance measurements remain pending; Guide 01 is in progress.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

**Goal:** Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on NSC Part 6 using the NVIDIA L4 Brev instance. This small POC follows **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**.

Follow these ten stages from node readiness to final evaluation. Each link gives the commands and evidence; training/validation can repeat during development.

1. [Node resources](docs/guides/02-nemotron-streaming-singapore-english.md#1-node-resources) — check disk, host RAM, and GPU availability.
2. [NeMo container](docs/guides/02-nemotron-streaming-singapore-english.md#2-nemo-container) — check the pinned Speech container's NeMo version and imports.
3. [Pretrained model](docs/guides/02-nemotron-streaming-singapore-english.md#3-pretrained-model) — restore Nemotron with persistent caching and confirm GPU placement.
4. [Data strategy and source checks](docs/guides/02-nemotron-streaming-singapore-english.md#4-data-strategy-and-source-checks) — define data roles and verify records, split separation, and annotation tags.
5. [Training and validation inputs](docs/guides/02-nemotron-streaming-singapore-english.md#5-training-and-validation-inputs) — build deterministic subsets, normalize text, and verify NeMo audio paths and language fields.
6. [Pretrained inference and baseline](docs/guides/02-nemotron-streaming-singapore-english.md#6-pretrained-inference-and-baseline) — verify one transcription, then establish offline WER on fixed validation.
7. [Training smoke test and artifacts](docs/guides/02-nemotron-streaming-singapore-english.md#7-training-smoke-test-and-artifacts) — run the official GPU smoke test and verify persistent checkpoint files.
8. [Checkpoint comparison and development](docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development) — restore and compare on fixed validation; iterate longer training and model selection.
9. [Held-out NSC test](docs/guides/02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test) — freeze model, decoding, and scoring choices before evaluating `nsc_test`.
10. [External benchmark](docs/guides/02-nemotron-streaming-singapore-english.md#10-external-benchmark) — check `gigaspeech_test` for generalization/regressions after the NSC test, without routine tuning.

**Current position (October 14, 2026):** Two-step smoke training/export evaluation, the configuration/data audits, the **100-step pilot**, and the frozen **held-out NSC test** are complete from operator evidence. Development WER was **10.84% pretrained**, **10.97% smoke**, and **10.47% pilot** on 50 recordings / 793 reference words. On the finalized held-out population of **3,670 recordings / 64,490 reference words**, pretrained scored **12.5663% WER** and POC v1 scored **12.2546% WER**: **201 fewer word errors**, **0.3117 percentage points absolute**, and **2.48% relative WER reduction**. The candidate therefore shows a modest, directionally consistent offline in-domain improvement; statistical significance, business-critical error quality, streaming behavior, performance, and production suitability remain unproven.

**Next decision — stage 10 is optional:** The core POC evidence is complete enough to pause. If broader regression evidence is valuable, prepare and run the frozen pretrained/POC v1 comparison on `gigaspeech_test`; do not tune from the held-out NSC result. Before any larger training spend, resolve pilot validation history, Lhotse exposure, loss, resources, and best/last/export provenance. The operator-supplied [evaluator](scripts/guide-02/baseline_eval.py) remains unchanged. See [current progress](PROGRESS.md#guide-02-end-of-day-checkpoint) and [completion boundaries](docs/guides/02-nemotron-streaming-singapore-english.md#completion-boundaries-and-next-step).

### Guide 04 — Single-node Slurm learning lab

[Open Guide 04](docs/guides/04-single-node-slurm-learning-lab.md)

**Goal:** Learn Slurm by using it on the existing NVIDIA L4 node. Read cluster state, understand the three small configuration files, submit CPU and GPU work, observe resource contention, inspect logs and job states, and troubleshoot controlled failures.

1. [Read cluster state](docs/guides/04-single-node-slurm-learning-lab.md#1-read-cluster-state) — understand partitions, nodes, resources, and state.
2. [Read the configuration](docs/guides/04-single-node-slurm-learning-lab.md#2-read-the-configuration) — understand the roles and change boundaries of the three active configuration files.
3. [Submit a CPU batch job](docs/guides/04-single-node-slurm-learning-lab.md#3-cpu-batch-job) and inspect its queue state and output.
4. [Use `srun`](docs/guides/04-single-node-slurm-learning-lab.md#4-interactive-srun) for an interactive scheduled task.
5. [Request and verify](docs/guides/04-single-node-slurm-learning-lab.md#5-gpu-allocation) the single NVIDIA L4 allocation.
6. [Observe contention](docs/guides/04-single-node-slurm-learning-lab.md#6-gpu-contention) while one job holds the only GPU and another waits.
7. [Control the lifecycle](docs/guides/04-single-node-slurm-learning-lab.md#7-lifecycle-control--deferred) with cancellation and timeout exercises; currently deferred.
8. [Run an existing Nemotron workload](docs/guides/04-single-node-slurm-learning-lab.md#8-scheduled-nemotron-workload) behind Slurm without modifying the existing inference helper; complete.

**Current position (October 15, 2026):** Slurm `26.05.5`, MUNGE, the controller, and the worker are operational. CPU batch submission, an interactive CPU shell, explicit allocation of `gpu:nvidia_l4:1`, two-job GPU contention, and a scheduled one-record Nemotron inference have been exercised from operator evidence. The functional path from `sbatch` through Docker, NeMo, and L4 inference is complete for this single-node lab. Cancellation/timeout, persistent accounting, and multi-GPU-aware container integration are deferred follow-up work.

## Future guides

Storage/data access → distributed training/NCCL → Kubernetes/GPU Operator → observability → architecture synthesis. These topics are planned in the [roadmap](ROADMAP.md); Guide 04 has reached its functional single-node workload checkpoint.

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
│       ├── baseline_eval.py
│       └── run_nsc_train.sh
└── docs/
    └── guides/
        ├── README.md         # Short directory index
        ├── 01-brev-gpu-node-validation.md
        ├── 02-nemotron-streaming-singapore-english.md
        └── 04-single-node-slurm-learning-lab.md
```

The [written guides](docs/guides/README.md) contain commands, architecture explanations, and recorded evidence. Runnable assets live under `scripts/guide-NN/`; the [Guide 02 tooling index](scripts/guide-02/README.md) explains six preparation helpers, the one-record GPU inference script, shared offline evaluator, and official-training launcher. The inference script implements the reported workflow; the training launcher is reconstructed from supplied command history, with the October 10 operator-supplied pilot case added. Neither has been compared byte-for-byte with the Brev copy or independently GPU-tested here. Repository scripts and the Brev workspace are separate locations; a commit does not deploy a helper to the node. The [progress log](PROGRESS.md) separates current status from historical milestones; the [roadmap](ROADMAP.md) holds future topics.

## Documentation standard

Distinguish observed results from prepared instructions and planned work. Record the environment, reproduction steps, troubleshooting lessons, and costs/cleanup where known. Publish reviewed, sanitized evidence; keep credentials, datasets, model caches/weights, raw recordings, and raw workload outputs outside Git. A local laptop path is not automatically available on an SSH-connected GPU host.

This is an independent learning guide and portfolio.
