# Learning Roadmap

The [repository README](README.md#guides-and-agendas) gives the agendas for the two active guides: GPU Node Validation and the Nemotron ASR POC. See [progress](PROGRESS.md) for recorded results.

This roadmap covers future topics. Milestones 03–08 are planned; their numbers continue the guide sequence and are separate from Guide 01's eight stack layers and Guide 02's ten execution stages. Add each guide and supporting files when its work begins. The order can evolve with the learning journey.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 03 — Storage and data access | Explore cloud and local storage, data movement, capacity, and I/O behavior. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| 04 — Slurm | Configure a suitable learning environment, submit GPU jobs, and inspect scheduling, allocation, logs, and failures. | Configuration, job scripts, observed job states, and troubleshooting notes. |
| 05 — Distributed training and communication | Explore collectives, accelerator communication, network topology, and distributed PyTorch/NeMo workloads, including NCCL where applicable. | Actual hardware topology, launch configuration, communication measurements, and limitations. |
| 06 — Kubernetes and GPU orchestration | Establish a GPU-enabled cluster and examine device allocation, operators, and workload lifecycle, including NVIDIA GPU Operator where applicable. | Manifests, operator configuration, environment details, and deployment validation. |
| 07 — Observability and troubleshooting | Inspect GPU and workload metrics, correlate logs, and investigate controlled failure scenarios. | Monitoring configuration, observed symptoms, diagnostic steps, and recovery evidence. |
| 08 — Architecture synthesis | Connect compute, scheduling, networking, and storage requirements into a documented AI infrastructure design. | Architecture diagram, tradeoff analysis, reproducible benchmark method, and clearly bounded conclusions. |

For the active ASR POC, continue with [Guide 02 stage 8 — checkpoint comparison and development](docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development). The two-step export evaluation and baseline recheck are complete: 10.97% versus 10.84% offline WER. Configuration and metadata duration audits are complete (300 eligible records / 0.656 h). The 100-step pilot completed and its export scored 10.47% offline WER, three fewer errors than pretrained (3.49% relative reduction). Recording-level comparison found 9 improved / 7 worsened / 34 equal-error-count recordings. Next: investigate validation history, Lhotse sampling/data exposure, training loss, and checkpoint/export provenance before deciding on further 300-/600-step training, data changes, or model selection. Held-out NSC testing (stage 9) and GigaSpeech evaluation (stage 10) follow finalized development choices. Streaming/performance, workload packaging, and Slurm submission are later work.

## Milestone completion criteria

- The experiment has been performed in an identified environment.
- The write-up explains how to reproduce it and links to relevant code or configuration.
- Results are supported by sanitized evidence and include limitations.
- Costs and cleanup steps are recorded when available.
- [PROGRESS.md](PROGRESS.md) is updated with the actual outcome and next step.
