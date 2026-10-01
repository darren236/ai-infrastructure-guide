# Learning Roadmap

All milestones below are **planned**. Completion will be recorded only after hands-on work and supporting evidence are available. The sequence can be adjusted as the learning journey develops.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 01 — Cloud GPU and CUDA basics | Inspect GPU hardware, driver, memory, CUDA compatibility, and basic utilization. | Sanitized environment inventory, command output, and explanation of the GPU software stack. |
| 02 — Containers and training | Establish a reproducible container environment and run a small PyTorch training workload; explore NeMo workflows. | Container definition, runnable commands, configuration, and observed training behavior. |
| 03 — Storage and data access | Explore cloud and local storage, data movement, capacity, and I/O behavior. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| 04 — Slurm | Configure a suitable lab environment, submit GPU jobs, and inspect scheduling, allocation, logs, and failures. | Configuration, job scripts, observed job states, and troubleshooting notes. |
| 05 — NCCL and distributed training | Explore collectives, GPU communication, network topology, and distributed PyTorch/NeMo workloads. | Actual hardware topology, launch configuration, communication measurements, and limitations. |
| 06 — Kubernetes and NVIDIA GPU Operator | Establish a GPU-enabled cluster and examine device allocation and workload lifecycle. | Manifests, operator configuration, environment details, and deployment validation. |
| 07 — Observability and troubleshooting | Inspect GPU and workload metrics, correlate logs, and investigate controlled failure scenarios. | Monitoring configuration, observed symptoms, diagnostic steps, and recovery evidence. |
| 08 — Architecture synthesis | Connect compute, scheduling, networking, and storage requirements into a documented AI infrastructure design. | Architecture diagram, tradeoff analysis, reproducible benchmark method, and clearly bounded conclusions. |

## Milestone completion criteria

- The experiment has been performed in an identified environment.
- The write-up explains how to reproduce it and links to relevant code or configuration.
- Results are supported by sanitized evidence and include limitations.
- Costs and cleanup steps are recorded when available.
- [PROGRESS.md](PROGRESS.md) is updated with the actual outcome and next step.
