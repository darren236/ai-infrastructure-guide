# Learning Roadmap

Guides 01 and 02 are in progress; milestones 03–08 are planned. See the [guide index](docs/guides/README.md) for navigation and [progress log](PROGRESS.md) for dated results. The sequence can evolve with the learning journey.

The [eight-layer agenda](README.md#agenda-from-physical-gpu-to-application) is the reading order for stack validation: physical GPU → PCIe → kernel → NVIDIA driver → CUDA runtime/Toolkit → container runtime → PyTorch/NeMo → application. The milestones below describe longer-term work; their numbers are separate from the agenda's layer numbers.

**Current next step:** Create deterministic Guide 02 subsets of approximately 300 training and 50 dev utterances. Preprocessing, baseline inference, training, and evaluation remain pending.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 01 — Compute and accelerator foundations (in progress) | Inspect cloud compute, accelerator hardware, drivers, runtimes, memory, and basic utilization. | [Layers 1–7 checks documented](docs/guides/01-brev-gpu-node-validation.md): host/driver checks, Toolkit discovery, architecture, container GPU access, devel-image compiler inspection, PyTorch CUDA availability/device identity, and linked NeMo model-loading checks. Actual GPU computation pending. |
| 02 — Nemotron streaming ASR POC (in progress) | Adapt NVIDIA Nemotron 3.5 Streaming ASR to Singapore English through a small NeMo POC/tutorial. | [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md): preflight, NeMo/model loading, and NSC downloads complete; tiny subsets, preprocessing, baseline inference, training, and evaluation pending. |
| 03 — Storage and data access | Explore cloud and local storage, data movement, capacity, and I/O behavior. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| 04 — Slurm | Configure a suitable learning environment, submit GPU jobs, and inspect scheduling, allocation, logs, and failures. | Configuration, job scripts, observed job states, and troubleshooting notes. |
| 05 — Distributed training and communication | Explore collectives, accelerator communication, network topology, and distributed PyTorch/NeMo workloads, including NCCL where applicable. | Actual hardware topology, launch configuration, communication measurements, and limitations. |
| 06 — Kubernetes and GPU orchestration | Establish a GPU-enabled cluster and examine device allocation, operators, and workload lifecycle, including NVIDIA GPU Operator where applicable. | Manifests, operator configuration, environment details, and deployment validation. |
| 07 — Observability and troubleshooting | Inspect GPU and workload metrics, correlate logs, and investigate controlled failure scenarios. | Monitoring configuration, observed symptoms, diagnostic steps, and recovery evidence. |
| 08 — Architecture synthesis | Connect compute, scheduling, networking, and storage requirements into a documented AI infrastructure design. | Architecture diagram, tradeoff analysis, reproducible benchmark method, and clearly bounded conclusions. |

## Milestone completion criteria

- The experiment has been performed in an identified environment.
- The write-up explains how to reproduce it and links to relevant code or configuration.
- Results are supported by sanitized evidence and include limitations.
- Costs and cleanup steps are recorded when available.
- [PROGRESS.md](PROGRESS.md) is updated with the actual outcome and next step.
