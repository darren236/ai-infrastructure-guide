# AI Infrastructure Guide

Practical AI infrastructure guides and a technical portfolio demonstrating NVIDIA Solutions Architect skills: GPU stack validation, containerized workloads, troubleshooting, and clear explanations of architecture and operational tradeoffs.

## Start here

| Guide | Focus | Current progress |
| --- | --- | --- |
| [01 — GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) | Brev access, Linux, NVIDIA drivers, CUDA containers, and PyTorch | Host and container GPU access, compiler inspection, and PyTorch CUDA availability validated. GPU computation remains pending. |
| [02 — Nemotron Streaming ASR for Singapore English](docs/guides/02-nemotron-streaming-singapore-english.md) | A small NeMo speech adaptation POC/tutorial | NeMo 3.0.0, model loading on `cuda:0`, persistent cache, and NSC query/dev downloads validated. Preprocessing, inference, and training remain pending. |

**Current next step:** Create deterministic Guide 02 subsets of roughly **300 training / 50 dev utterances**. No completed baseline inference, fine-tuning, evaluation, or benchmark is claimed.

Browse the [guide index](docs/guides/README.md), check the [progress log](PROGRESS.md) for recorded milestones, or follow the [roadmap](ROADMAP.md) for future work.

## Repository layout

```text
ai-infrastructure-guide/
├── README.md                 # Entry point and current status
├── ROADMAP.md                # Planned learning sequence
├── PROGRESS.md               # Dated milestones and evidence
├── .gitignore
├── docs/
│   └── guides/
│       ├── README.md         # Guide index
│       ├── 01-brev-gpu-node-validation.md
│       └── 02-nemotron-streaming-singapore-english.md
└── guides/
    └── README.md             # Placement of future runnable guide assets
```

Written instructions live in `docs/guides/`. Add code and configuration under `guides/` when a guide needs reusable files; directory conventions are explained [here](guides/README.md). Future work includes Slurm, NCCL, Kubernetes, NVIDIA GPU Operator, observability, networking, and storage. Create supporting directories when there is content to include.

## Documentation standard

Each guide records its objective, actual environment, reproduction steps, observed results, troubleshooting lessons, and costs/cleanup where known. Distinguish completed checks from planned work and describe the limits of each result. Publish reviewed, sanitized evidence; keep credentials, datasets, model weights, raw terminal recordings, and generated outputs outside Git.

This is an independent learning guide and portfolio.
