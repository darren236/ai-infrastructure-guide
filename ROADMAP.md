# Further Learning

The three [published guides](README.md#guides-and-agendas) are complete for their stated core outcomes. This page collects optional extensions and broader topics without assigning guide numbers before a guide is published.

## Optional extensions to the published guides

| Guide | What to try | Why it matters | Where the evidence stops |
| --- | --- | --- | --- |
| [Guide 01 — GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) | Run a standalone CUDA micro-test, exercise distributed execution, and measure latency, throughput, utilization, and thermals. | These checks extend functional access into compiler, communication, and performance evidence. | The recorded work verifies the path through a real application, but not isolated kernel behavior, multi-GPU communication, or sustained performance. |
| [Guide 02 — Nemotron ASR Adaptation](docs/guides/02-nemotron-streaming-singapore-english.md) | Run the prepared full-data workflow on a 32 GiB-or-larger node, compare native streaming behavior, and evaluate GigaSpeech without tuning from it. | These exercises test resource capacity, deployment-mode quality, and out-of-domain generalization. | The 300/50 POC is complete. The 2,152/1,203 workflow is prepared but has no new candidate result; streaming and GigaSpeech comparisons have not been run. |
| [Guide 03 — Scheduling GPU Workloads with Slurm](docs/guides/03-scheduling-gpu-workloads-with-slurm.md) | Exercise cancellation and timeouts, configure persistent accounting, and test scheduler-aware container binding on more than one GPU. | These exercises add lifecycle history, operational visibility, and stronger isolation evidence. | The recorded single-node path ends with scheduled one-record Nemotron inference. |

The optional Guide 02 full-data workflow preserves NeMo Speech v3.0.0 and the official training entry point. Two 10 GiB-container attempts on the original node reached five steps and ended in confirmed memory-cgroup OOMs. The next evidence gate is duration/OOM profiling and an independent 50-step resource pilot on a 32 GiB-or-larger node. Full training should continue only if that gate passes.

## Broader topics

| Topic | Why explore it | Evidence to add |
| --- | --- | --- |
| Storage and data access | Connect capacity, movement, locality, and I/O behavior to model workflows. | Storage architecture, reproducible checks, measured observations, and cleanup notes. |
| Distributed training and communication | Understand collectives, accelerator communication, topology, and NCCL behavior where applicable. | Hardware topology, launch configuration, communication measurements, and limitations. |
| Kubernetes and GPU orchestration | Compare scheduler and operator approaches to accelerator allocation and workload lifecycle. | Manifests, operator configuration, environment details, and deployment validation. |
| Observability and troubleshooting | Correlate GPU metrics, workload logs, and controlled failure symptoms. | Monitoring configuration, observed symptoms, diagnostic steps, and recovery evidence. |
| Architecture synthesis | Connect compute, scheduling, networking, storage, and model requirements. | Architecture diagrams, tradeoff analysis, reproducible benchmark methods, and bounded conclusions. |

## Evidence standard for future additions

- Perform the exercise in an identified environment.
- Explain how to reproduce it and link to relevant code or configuration.
- Support results with reviewed, sanitized evidence and state the limitations.
- Record costs and cleanup steps when available.
- Update the [Learning Record](PROGRESS.md) with the actual outcome and the next useful question.

Topics receive guide numbers only when their core learning journey and supporting evidence are ready to publish.
