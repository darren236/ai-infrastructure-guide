# Learning Roadmap

The [repository README](README.md#guides-and-agendas) gives the agendas for the active guides: GPU Node Validation, the Nemotron ASR POC, and the single-node Slurm learning lab. See [progress](PROGRESS.md) for recorded results.

This roadmap covers the broader sequence. Milestone 04 is active; milestones 03 and 05–08 remain planned. Their numbers continue the guide sequence and are separate from the stages inside each guide. Add supporting files when actual work needs them. The order can evolve with the learning journey.

| Milestone | Focus | Evidence to produce |
| --- | --- | --- |
| 03 — Storage and data access | Explore cloud and local storage, data movement, capacity, and I/O behavior. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| [04 — Slurm (active)](docs/guides/04-single-node-slurm-learning-lab.md) | Use a functional single-node learning environment to submit CPU/GPU work and inspect scheduling, allocation, logs, contention, and failures. | Initial configuration and idle-node evidence recorded; job scripts, observed job states, and workload troubleshooting remain next. |
| 05 — Distributed training and communication | Explore collectives, accelerator communication, network topology, and distributed PyTorch/NeMo workloads, including NCCL where applicable. | Actual hardware topology, launch configuration, communication measurements, and limitations. |
| 06 — Kubernetes and GPU orchestration | Establish a GPU-enabled cluster and examine device allocation, operators, and workload lifecycle, including NVIDIA GPU Operator where applicable. | Manifests, operator configuration, environment details, and deployment validation. |
| 07 — Observability and troubleshooting | Inspect GPU and workload metrics, correlate logs, and investigate controlled failure scenarios. | Monitoring configuration, observed symptoms, diagnostic steps, and recovery evidence. |
| 08 — Architecture synthesis | Connect compute, scheduling, networking, and storage requirements into a documented AI infrastructure design. | Architecture diagram, tradeoff analysis, reproducible benchmark method, and clearly bounded conclusions. |

For the active ASR POC, [Guide 02 stage 9 — held-out NSC test](docs/guides/02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test) is complete. On 3,670 compatible held-out recordings / 64,490 reference words, the frozen 100-step POC v1 reduced WER from 12.5663% to 12.2546%: 201 fewer word errors, 0.3117 percentage points absolute, and 2.48% relative. This supports a modest offline in-domain generalization claim, not production readiness. The POC may pause here; [stage 10](docs/guides/02-nemotron-streaming-singapore-english.md#10-external-benchmark) GigaSpeech evaluation is an optional frozen-model regression check if broader evidence is worth the additional work. Validation history, Lhotse exposure, training loss, resource evidence, and checkpoint/export provenance remain limitations to resolve before any further 300-/600-step training. Streaming/performance, workload packaging, and Slurm submission are later work.

## Milestone completion criteria

- The experiment has been performed in an identified environment.
- The write-up explains how to reproduce it and links to relevant code or configuration.
- Results are supported by sanitized evidence and include limitations.
- Costs and cleanup steps are recorded when available.
- [PROGRESS.md](PROGRESS.md) is updated with the actual outcome and next step.
