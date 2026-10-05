# Learning Roadmap

Milestone 01 is **in progress**: host access, Linux identification, PCIe visibility, driver communication, host Toolkit discovery, and host/container architecture are documented in [Guide 01](docs/guides/01-brev-gpu-node-validation.md). Docker, NVIDIA container GPU access, and devel-image compiler inspection are complete for the reported checks. PyTorch CUDA availability and GPU identity are validated in an NGC container; actual GPU computation and the Guide 01 completion decision remain pending. Guide 02 is also **in progress**: NeMo model loading on the L4 and NSC query/dev downloads are recorded in [the ASR guide](docs/guides/02-nemotron-streaming-singapore-english.md). Tiny subsets, preprocessing, baseline inference, training, and evaluation remain pending. Milestones 03–08 are planned; the sequence can be adjusted as the learning journey develops.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 01 — Compute and accelerator foundations (in progress) | Inspect cloud compute, accelerator hardware, drivers, runtimes, memory, and basic utilization. | [Parts 1–4 documented](docs/guides/01-brev-gpu-node-validation.md): host/driver checks, Toolkit discovery, architecture, container GPU access, devel-image compiler inspection, and PyTorch CUDA availability/device identity. Actual GPU computation pending. |
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
