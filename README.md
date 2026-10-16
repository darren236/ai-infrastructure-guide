# AI Infrastructure Guide

This repository shares what we learned while moving an NVIDIA GPU workload from node validation, through model adaptation and evaluation, to scheduled execution. Each guide connects a practical question to the checks we ran, the evidence we recorded, and the boundary beyond which the evidence stops.

The three published guides form one complete learning path:

1. Validate the GPU node and identify which software layer answers each troubleshooting question.
2. Adapt and evaluate a streaming ASR model without mixing training, development selection, and held-out testing.
3. Schedule CPU, GPU, and application workloads with Slurm and observe how resource allocation behaves.

## Start here

- Start with [Guide 01 — GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) to understand the GPU software stack from hardware to application.
- Continue with [Guide 02 — Nemotron ASR Adaptation](docs/guides/02-nemotron-streaming-singapore-english.md) to follow a complete ASR proof of concept and its evaluation boundaries.
- Finish with [Guide 03 — Scheduling GPU Workloads with Slurm](docs/guides/03-scheduling-gpu-workloads-with-slurm.md) to see how CPU, GPU, and application work is allocated through a scheduler.
- Use the [Learning Record](PROGRESS.md#current-state) for current outcomes and dated experiment history.
- Use [Further Learning](ROADMAP.md) for optional extensions and future topics.
- Use the [Guide 02 tooling index](scripts/guide-02/README.md) for helper scripts and the optional full-data runbook.

Recommended order: **Guide 01 → Guide 02 → Guide 03**. Each guide reaches its stated core outcome. Exercises that have not been run are collected under further learning rather than presented as missing chapters.

<a id="guides-and-agendas"></a>
<a id="agenda-from-physical-gpu-to-application"></a>
<a id="guides"></a>

## Guides and what we learned

### Guide 01 — GPU Node Validation

[Open Guide 01](docs/guides/01-brev-gpu-node-validation.md)

**Goal:** Understand the path from the provisioned GPU to a working application on an NVIDIA L4 Brev node. Follow these eight layers in order, or jump to the minimum check that answers the current troubleshooting question.

1. [Physical GPU](docs/guides/01-brev-gpu-node-validation.md#1-physical-gpu) — identify the provisioned GPU and the boundary of remote inventory evidence.
2. [PCIe enumeration](docs/guides/01-brev-gpu-node-validation.md#2-pcie-enumeration) — check whether Linux can see the NVIDIA device with `lspci`.
3. [Linux kernel](docs/guides/01-brev-gpu-node-validation.md#3-linux-kernel) — identify the operating system, kernel, architecture, and execution context.
4. [NVIDIA driver](docs/guides/01-brev-gpu-node-validation.md#4-nvidia-driver) — use `nvidia-smi`; its CUDA field reports driver compatibility, not an installed Toolkit version.
5. [CUDA runtime / Toolkit](docs/guides/01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit) — inspect `nvcc` in a development container; this setup does not require a host Toolkit.
6. [Container runtime](docs/guides/01-brev-gpu-node-validation.md#6-container-runtime) — validate Docker, NVIDIA integration, and GPU-enabled container execution.
7. [PyTorch / NeMo](docs/guides/01-brev-gpu-node-validation.md#7-pytorch--nemo) — verify framework CUDA access, GPU identity, NeMo import, and model placement.
8. [Application](docs/guides/01-brev-gpu-node-validation.md#8-application) — process real inputs and verify the output through the Guide 02 workflow.

**What we learned:** The recorded L4 path works from host visibility through a real ASR application. Each check proves only its own layer and scope; it does not establish performance or production readiness. A standalone CUDA micro-test, distributed execution, and performance measurements are optional extensions.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

**Goal:** Adapt `nvidia/nemotron-3.5-asr-streaming-0.6b` to Singapore English while keeping training, development selection, and held-out evaluation separate.

1. [Node resources](docs/guides/02-nemotron-streaming-singapore-english.md#1-node-resources) — check disk, host RAM, and GPU availability before pulling models or data.
2. [NeMo container](docs/guides/02-nemotron-streaming-singapore-english.md#2-nemo-container) — verify the pinned Speech container, NeMo version, and imports.
3. [Pretrained model](docs/guides/02-nemotron-streaming-singapore-english.md#3-pretrained-model) — restore Nemotron with persistent caching and confirm GPU placement.
4. [Data strategy and source checks](docs/guides/02-nemotron-streaming-singapore-english.md#4-data-strategy-and-source-checks) — define split roles and verify records, speaker separation, audio mapping, and annotation tags.
5. [Training and validation inputs](docs/guides/02-nemotron-streaming-singapore-english.md#5-training-and-validation-inputs) — build deterministic 300/50 subsets, normalize declared tags, and create verified NeMo manifests.
6. [Pretrained inference and baseline](docs/guides/02-nemotron-streaming-singapore-english.md#6-pretrained-inference-and-baseline) — verify one transcription, then establish the fixed-development baseline of 10.84% offline WER.
7. [Training smoke test and artifacts](docs/guides/02-nemotron-streaming-singapore-english.md#7-training-smoke-test-and-artifacts) — run the official NeMo path and confirm that checkpoints and an export persist.
8. [Checkpoint comparison and development](docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development) — restore exports, compare them on fixed development data, and freeze POC v1 after the 100-step pilot reached 10.47% WER.
9. [Held-out NSC test](docs/guides/02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test) — evaluate the frozen candidate once under the declared policy: 12.5663% to 12.2546% WER, or 201 fewer errors over 64,490 words.

**What we learned:** The nine-stage historical 300/50 proof of concept is complete through frozen held-out evaluation. Its 2.48% relative WER reduction is modest offline in-domain evidence, not a production claim. Full-data fine-tuning, native streaming comparison, and GigaSpeech evaluation remain optional extensions.

The separate [full-data workflow](scripts/guide-02/full-finetune/README.md) is prepared for 2,152 train and 1,203 development recordings. Two attempts in 10 GiB containers reached five steps and ended in confirmed memory-cgroup OOMs. Its next gate is a 32 GiB-or-larger node and an independent 50-step resource pilot; no new candidate result is claimed.

### Guide 03 — Scheduling GPU Workloads with Slurm

[Open Guide 03](docs/guides/03-scheduling-gpu-workloads-with-slurm.md)

**Goal:** Learn how Slurm represents resources, decides whether work can run, and launches an existing GPU application.

1. [Read cluster state](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#1-read-cluster-state) — use `sinfo` and `scontrol` to understand partitions, nodes, resources, and state.
2. [Read the configuration](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#2-read-the-configuration) — understand the separate roles of `slurm.conf`, `gres.conf`, and `cgroup.conf`.
3. [Submit a CPU batch job](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#3-cpu-batch-job) — follow a job from `sbatch` submission through queue state to its output file.
4. [Run interactively](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#4-interactive-srun) — use `srun` to distinguish login-shell work from scheduled work.
5. [Submit a GPU job](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#5-gpu-allocation) — request the L4 and verify its scheduler-side allocation.
6. [Observe contention](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#6-gpu-contention) — hold the only GPU and see why a second request waits for `Resources`.
7. [Schedule the existing workload](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#7-scheduled-nemotron-workload) — run the existing one-record Nemotron helper through `sbatch` without modifying it.

**What we learned:** Scheduler-visible resources, allocations, and application utilization are different questions. The seven-step path is verified through scheduled Nemotron inference; cancellation and timeout exercises, persistent accounting, and multi-GPU-aware container integration remain optional extensions.

## Repository map

```text
ai-infrastructure-guide/
├── README.md                         # Shared learning path and guide outcomes
├── PROGRESS.md                       # Current outcomes and dated learning record
├── ROADMAP.md                        # Optional extensions and further learning
├── docs/
│   ├── guides/
│   │   ├── 01-brev-gpu-node-validation.md
│   │   ├── 02-nemotron-streaming-singapore-english.md
│   │   └── 03-scheduling-gpu-workloads-with-slurm.md
│   └── evidence/                     # Reviewed, sanitized evidence suitable for Git
└── scripts/
    └── guide-02/
        ├── README.md                 # Helper index and historical POC notes
        ├── *.py / run_nsc_train.sh  # Preparation, inference, evaluation, and training helpers
        └── full-finetune/            # Prepared optional full-data workflow
```

The guides explain what to run, why it matters, and what each result establishes. Scripts are repository assets, not proof of execution: committing a helper does not deploy it to a GPU node.

## Evidence, reproducibility, and data safety

- Treat results as verified only within the recorded environment and scope.
- Distinguish recorded observations from prepared instructions and optional exercises that have not been run.
- Keep credentials, raw datasets, audio, model caches, weights, checkpoints, predictions, and unreviewed logs outside Git.
- Commit only reviewed, sanitized evidence with deliberate filenames under `docs/evidence/`.
- Record environment versions, reproduction steps, scoring policy, limitations, and cleanup or cost information when available.
- Treat local repository paths and remote GPU-host paths as separate environments unless deployment is explicitly performed.

The evidence supports bounded learning outcomes, not universal compatibility, statistical significance, sustained performance, or production readiness. See [Further Learning](ROADMAP.md) for optional extensions and future topics.
