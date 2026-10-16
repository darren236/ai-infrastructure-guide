# AI Infrastructure Guide

This repository shares what we learned while moving an NVIDIA GPU workload from node validation, through model adaptation and evaluation, to scheduled execution. Each guide connects a practical question to the checks we ran, the evidence we recorded, and the boundary beyond which the evidence stops.

The three published guides form one complete learning path:

1. Validate the GPU node and identify which software layer answers each troubleshooting question.
2. Adapt and evaluate a streaming ASR model without mixing training, development selection, and held-out testing.
3. Schedule CPU, GPU, and application workloads with Slurm and observe how resource allocation behaves.

## Start here

| If you want to... | Start with |
| --- | --- |
| Understand the GPU software stack from hardware to application | [Guide 01 — GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) |
| Follow a complete ASR adaptation proof of concept and its evaluation boundaries | [Guide 02 — Nemotron ASR Adaptation](docs/guides/02-nemotron-streaming-singapore-english.md) |
| Learn how CPU and GPU work is allocated through Slurm | [Guide 03 — Scheduling GPU Workloads with Slurm](docs/guides/03-scheduling-gpu-workloads-with-slurm.md) |
| Review current outcomes and the dated experiment history | [Learning Record](PROGRESS.md#current-state) |
| Explore optional follow-on work | [Further Learning](ROADMAP.md) |
| Find Guide 02 helpers and the optional full-data runbook | [Guide 02 tooling index](scripts/guide-02/README.md) |

Recommended order: **Guide 01 → Guide 02 → Guide 03**. Each guide reaches its stated core outcome. Exercises that have not been run are collected under further learning rather than presented as missing chapters.

<a id="guides-and-agendas"></a>
<a id="agenda-from-physical-gpu-to-application"></a>
<a id="guides"></a>

## Guides and what we learned

### Guide 01 — GPU Node Validation

[Open Guide 01](docs/guides/01-brev-gpu-node-validation.md)

| Question | What we did | What we learned |
| --- | --- | --- |
| What GPU and host were provisioned? | Checked remote inventory, PCIe enumeration, operating system, kernel, and architecture. | The recorded node was Ubuntu 22.04 on x86_64 with an NVIDIA L4 in the 24 GB class. |
| Can the NVIDIA driver communicate with the GPU? | Inspected the host with `nvidia-smi`. | The driver saw the L4. The CUDA value shown by `nvidia-smi` is a driver compatibility ceiling, not proof that a host CUDA Toolkit is installed. |
| Where should CUDA tools and libraries live? | Checked the host, then inspected CUDA runtime and development containers. | The checked host had no Toolkit in the expected locations; the development container supplied `nvcc` and the userspace stack. |
| Can containers and frameworks use the GPU? | Validated Docker/NVIDIA integration, PyTorch CUDA access, and NeMo model restoration. | The path from container runtime through PyTorch and NeMo to `cuda:0` worked in the recorded environment. |
| Does a real application complete? | Followed the Guide 02 path through inference, training, export restoration, and held-out evaluation. | Application evidence answers more than an isolated device check, while still not proving performance or production readiness. |

**Core outcome:** The host-to-application access path is verified for the recorded L4 environment. A standalone CUDA micro-test, distributed execution, and performance measurements are optional extensions.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

| Question | What we did | What we learned |
| --- | --- | --- |
| Is the environment ready for the model and data? | Checked node resources, pinned the NeMo container, restored the pretrained model, and audited source data and split separation. | Environment, model, and data assumptions should be made explicit before training begins. |
| Can we build reproducible model inputs? | Created deterministic 300/50 subsets, normalized declared annotation tags, and converted them to NeMo manifests. | Data lineage and split roles matter as much as the training command. |
| What does the pretrained model establish? | Ran single-record inference and a fixed 50-record development baseline. | The baseline was 10.84% offline WER over 793 reference words and became the comparison point for later exports. |
| Does the official adaptation path work? | Completed the two-step smoke test, a controlled 100-step pilot, export restoration, and recording-level comparison. | The selected POC v1 export reached 10.47% offline development WER; a successful run still needs provenance and evaluation boundaries. |
| Does the frozen candidate improve on held-out NSC data? | Froze the candidate and scoring policy before evaluating 3,670 compatible recordings / 64,490 words. | WER changed from 12.5663% to 12.2546%: 201 fewer errors and a 2.48% relative reduction. This is modest offline in-domain evidence, not a production claim. |

**Core outcome:** The nine-stage historical 300/50 proof of concept is complete through frozen held-out evaluation. Full-data fine-tuning, native streaming comparison, and GigaSpeech evaluation remain optional extensions.

The separate [full-data workflow](scripts/guide-02/full-finetune/README.md) is prepared for 2,152 train and 1,203 development recordings. Two attempts in 10 GiB containers reached five steps and ended in confirmed memory-cgroup OOMs. Its next gate is a 32 GiB-or-larger node and an independent 50-step resource pilot; no new candidate result is claimed.

### Guide 03 — Scheduling GPU Workloads with Slurm

[Open Guide 03](docs/guides/03-scheduling-gpu-workloads-with-slurm.md)

| Question | What we did | What we learned |
| --- | --- | --- |
| How does Slurm represent the node? | Read cluster state and the configuration that defines CPU, memory, GPU, and cgroup behavior. | Scheduler-visible resources are configured abstractions and should be checked separately from physical inventory. |
| Can ordinary and interactive work run through the scheduler? | Submitted a CPU batch job and opened an interactive `srun` allocation. | Partition selection, CPU allocation, job state, and shell context can be observed directly. |
| Does Slurm allocate the configured GPU? | Requested the L4 and inspected scheduler-side allocation state. | The observed job received `gpu:nvidia_l4:1`; allocation is not the same as application utilization. |
| What happens when the GPU is already allocated? | Held the L4 in one job and submitted another GPU request. | The second job remained pending for `Resources`, demonstrating contention handling in the single-node case. |
| Can the existing application run behind Slurm? | Launched the one-record Nemotron inference helper through `sbatch`. | The complete path from Slurm through Docker and NeMo to L4 inference worked without changing the application helper. |

**Core outcome:** The seven-step path ends with verified scheduled Nemotron inference. Cancellation and timeout exercises, persistent accounting, and multi-GPU-aware container integration are optional extensions.

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
