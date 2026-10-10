# Guide index

Start with the [per-guide agendas in the repository README](../../README.md#guides-and-agendas), then open a guide for its detailed instructions and results.

| Guide | Purpose | Current checkpoint |
| --- | --- | --- |
| [01 — GPU Node Validation](01-brev-gpu-node-validation.md) | Follow the eight layers from physical GPU through driver, CUDA, container, framework, and application checks. | [Stack checks recorded](01-brev-gpu-node-validation.md#completion-boundaries-and-next-step); one ASR application inference is verified in Guide 02; standalone CUDA/tensor checks pending. |
| [02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](02-nemotron-streaming-singapore-english.md) | Follow ten checks from node resources and NeMo/model setup through data preparation, baseline inference, training, checkpoint comparison, held-out test, and external evaluation. | [Official two-step GPU training and artifact saving verified on October 9](02-nemotron-streaming-singapore-english.md#7-training-smoke-test-and-artifacts); [Stage 8 — restore/evaluate the export](02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development) is next. Pretrained baseline remains 793 reference words / 10.84% offline WER. |

Guide 02 follows the same [ten-stage workflow in the README](../../README.md#guide-02--adapting-nvidia-nemotron-35-streaming-asr-to-singapore-english) and [guide agenda](02-nemotron-streaming-singapore-english.md#hands-on-agenda), from node resources through model/data checks to training and evaluation. Its [section map](02-nemotron-streaming-singapore-english.md#guide-sections) and [tooling index](../../scripts/guide-02/README.md) provide detailed navigation.

Both guides are in progress. See their completion sections for the limits of recorded results, the [progress log](../../PROGRESS.md) for dated evidence, and the [roadmap](../../ROADMAP.md) for future topics.
