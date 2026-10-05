# Learning Roadmap

Milestone 01 is **in progress**: host access, Linux identification, PCIe visibility, driver communication, host Toolkit discovery, and host/container architecture are documented in [Lab 01](docs/labs/01-brev-gpu-node-validation.md). Docker, NVIDIA container GPU access, and devel-image compiler inspection are complete for the reported checks. Actual CUDA kernel execution remains pending. Milestones 02–08 are planned; the sequence can be adjusted as the learning journey develops.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 01 — Compute and accelerator foundations (in progress) | Inspect cloud compute, accelerator hardware, drivers, runtimes, memory, and basic utilization. | [Parts 1–3 documented](docs/labs/01-brev-gpu-node-validation.md): host/driver checks, Toolkit discovery, architecture, container GPU access, and devel-image compiler inspection. Actual CUDA kernel execution pending. |
| 02 — Containers and training | Establish a reproducible container environment and run a small PyTorch training workload; explore NeMo workflows. | Container definition, runnable commands, configuration, and observed training behavior. |
| 03 — Storage and data access | Explore cloud and local storage, data movement, capacity, and I/O behavior. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| 04 — Slurm | Configure a suitable lab environment, submit GPU jobs, and inspect scheduling, allocation, logs, and failures. | Configuration, job scripts, observed job states, and troubleshooting notes. |
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
