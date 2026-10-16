# Guide index

The guides form one sequential learning journey. Start with the [overview and shared agenda](../../README.md#guides-and-agendas), then use each guide for detailed commands, evidence, and boundaries.

| Guide | Question | What we did | What we learned |
| --- | --- | --- | --- |
| [01 — GPU Node Validation](01-brev-gpu-node-validation.md) | Which layer should answer a GPU troubleshooting question? | Followed the path from hardware and the Linux host through containers, PyTorch, NeMo, and application evidence. | The recorded L4 path is verified from host to application; a successful check proves only its stated layer and scope. |
| [02 — Nemotron ASR Adaptation](02-nemotron-streaming-singapore-english.md) | How can model adaptation keep training, development selection, and held-out testing distinct? | Completed a nine-stage 300/50 proof of concept through frozen held-out evaluation. | POC v1 reduced held-out WER from 12.5663% to 12.2546% over 3,670 recordings / 64,490 words, a modest offline in-domain result. |
| [03 — Scheduling GPU Workloads with Slurm](03-scheduling-gpu-workloads-with-slurm.md) | How does Slurm allocate CPU/GPU resources and launch an existing workload? | Read scheduler state, submitted CPU and GPU jobs, observed contention, and scheduled Nemotron inference. | The single-node path works through application execution; allocation, utilization, accounting, and production operations are separate concerns. |

Each core guide is complete for its stated learning outcome. Optional extensions—including a CUDA micro-test, full-data ASR training, native streaming comparison, GigaSpeech evaluation, lifecycle controls, accounting, and multi-GPU-aware container integration—are collected in [Further Learning](../../ROADMAP.md).

Guide 02 also provides a [section map](02-nemotron-streaming-singapore-english.md#guide-sections), [nine-stage agenda](02-nemotron-streaming-singapore-english.md#hands-on-agenda), [tooling index](../../scripts/guide-02/README.md), and [optional full-data runbook](../../scripts/guide-02/full-finetune/README.md).

See the [Learning Record](../../PROGRESS.md) for current outcomes and dated evidence.
