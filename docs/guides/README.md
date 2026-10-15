# Guide index

Start with the [per-guide agendas in the repository README](../../README.md#guides-and-agendas), then open a guide for its detailed instructions and results.

| Guide | Purpose | Current checkpoint |
| --- | --- | --- |
| [01 — GPU Node Validation](01-brev-gpu-node-validation.md) | Follow the eight layers from physical GPU through driver, CUDA, container, framework, and application checks. | [Stack checks recorded](01-brev-gpu-node-validation.md#completion-boundaries-and-next-step); one ASR application inference is verified in Guide 02; standalone CUDA/tensor checks pending. |
| [02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](02-nemotron-streaming-singapore-english.md) | Follow ten checks from node resources and NeMo/model setup through data preparation, baseline inference, training, checkpoint comparison, held-out test, and external evaluation. | [Stage 9](02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test) is complete: on 3,670 held-out recordings / 64,490 words, pretrained scored 12.5663% WER and the frozen POC v1 scored 12.2546%, a 2.48% relative reduction / 201 fewer errors. This is modest offline in-domain evidence, not production validation. [Stage 10](02-nemotron-streaming-singapore-english.md#10-external-benchmark) is optional; training-behavior/provenance limitations remain before any larger run. |
| [04 — Single-node Slurm learning lab](04-single-node-slurm-learning-lab.md) | Learn scheduler use through cluster inspection, CPU and GPU jobs, interactive work, contention, lifecycle control, and controlled failures. | CPU batch, interactive CPU, explicit L4 allocation, and two-job GPU contention are complete from operator evidence. Cancellation and timeout behavior are next. |

Guide 02 follows the same [ten-stage workflow in the README](../../README.md#guide-02--adapting-nvidia-nemotron-35-streaming-asr-to-singapore-english) and [guide agenda](02-nemotron-streaming-singapore-english.md#hands-on-agenda), from node resources through model/data checks to training and evaluation. Its [section map](02-nemotron-streaming-singapore-english.md#guide-sections) and [tooling index](../../scripts/guide-02/README.md) provide detailed navigation.

The guides are in progress. See their completion sections for the limits of recorded results, the [progress log](../../PROGRESS.md) for dated evidence, and the [roadmap](../../ROADMAP.md) for future topics.
