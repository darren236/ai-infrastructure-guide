# Guide index

Start with the [per-guide agendas in the repository README](../../README.md#guides-and-agendas), then open a guide for its detailed instructions and results.

| Guide | Purpose | Current checkpoint |
| --- | --- | --- |
| [01 — GPU Node Validation](01-brev-gpu-node-validation.md) | Follow the eight layers from physical GPU through driver, CUDA, container, framework, and application checks. | [Access and inspection checks recorded](01-brev-gpu-node-validation.md#completion-boundaries-and-next-step); actual GPU computation/application execution pending. |
| [02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](02-nemotron-streaming-singapore-english.md) | Follow a small ASR POC through train/validation, configuration freeze, NSC held-out test, and GigaSpeech external/OOD evaluation. | [POC subsets independently checked](02-nemotron-streaming-singapore-english.md#end-of-day-checkpoint); transcript normalization next. |

Guide 02 includes a [section map](02-nemotron-streaming-singapore-english.md#guide-sections) and a separate [tooling index](../../scripts/guide-02/README.md) for its reusable helpers.

Both guides are in progress. See their completion sections for the limits of recorded results, the [progress log](../../PROGRESS.md) for dated evidence, and the [roadmap](../../ROADMAP.md) for future topics.
