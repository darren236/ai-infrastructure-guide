# AI Infrastructure Guide

A hands-on guide to AI infrastructure and a technical portfolio covering compute, workload orchestration, distributed systems, and reliable operations. This repository will document reproducible guides, architecture decisions, operational lessons, and measured results as the work is completed.

The learning scope covers **GPU compute, containers, PyTorch/NeMo training, cloud infrastructure, Slurm, distributed communication, Kubernetes, observability, networking, storage, and troubleshooting**. Tool-specific guides will explore technologies such as CUDA, NCCL, and NVIDIA GPU Operator within that broader scope. The aim is to build practical experience designing and operating AI infrastructure and explain the tradeoffs clearly.

## Current status

**Guide 01, Part 1 complete:** [Brev GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) documents cloud connectivity, Linux host identification, PCIe GPU visibility, and NVIDIA driver communication on an L4 node.

**Guide 01:** [PyTorch CUDA access through an NVIDIA NGC container](docs/guides/01-brev-gpu-node-validation.md#part-4-pytorch-cuda-access-through-an-nvidia-ngc-container) documents PyTorch CUDA availability and NVIDIA L4 identity in the `26.09-py3` container, its reported forward-compatibility mode, and interactive/one-liner workflows.

**Latest milestone — Guide 02:** [Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](docs/guides/02-nemotron-streaming-singapore-english.md) records node preflight, NeMo 3.0.0 validation, model loading on `cuda:0`, persistent cache configuration, and NSC query/dev data downloads. Next: deterministic POC subsets of roughly 300 training and 50 dev utterances.

Actual GPU inference/computation, baseline evaluation, preprocessing, real training, distributed training/NCCL, and performance benchmarking remain pending. No training runs, deployments, benchmarks, or complete end-to-end guide are claimed.

Follow the planned milestones in [ROADMAP.md](ROADMAP.md) and the evidence-backed activity log in [PROGRESS.md](PROGRESS.md).

## Learning objectives

- Understand accelerator hardware, runtime and driver compatibility, resource usage, and failure modes.
- Run reproducible containerized NeMo/PyTorch workloads on cloud GPUs.
- Schedule compute jobs with Slurm and investigate distributed training and communication.
- Operate GPU workloads on Kubernetes and explore device management and operators.
- Evaluate observability, networking, and storage requirements, and document troubleshooting decisions.

## Repository layout

```text
ai-infrastructure-guide/
├── README.md
├── ROADMAP.md
├── PROGRESS.md
├── .gitignore
├── docs/             # Concepts, architecture notes, and operational guides
│   └── guides/       # Numbered guides and evidence
├── guides/           # Guide-specific code and configuration
├── scripts/          # Reusable setup, inspection, and validation scripts
├── containers/       # Container definitions and workload environments
├── slurm/            # Scheduler configuration and job scripts
├── kubernetes/       # Manifests and GPU platform configuration
├── benchmarks/       # Benchmark methods and verified measurements
└── diagrams/         # Architecture diagrams and editable sources
```

Empty directories contain `.gitkeep` placeholders so Git preserves the scaffold.

## Continue the guide

Start with [Guide 01: GPU Node Validation](docs/guides/01-brev-gpu-node-validation.md) for the environment inventory, connectivity recovery, host Toolkit discovery, and host/container architecture. Docker, NVIDIA container GPU access, and devel-image compiler inspection are documented in Part 3. PyTorch availability and device identity checks are documented in Part 4. Guide 01 remains in progress; a small GPU tensor computation with verified output is the recommended final functional check before closing Guide 01. Continue with [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) for the Nemotron ASR POC; its next planned step is tiny deterministic training/dev subsets.

## Documentation standard

Each completed guide should record:

1. The objective and architecture, including relevant tradeoffs.
2. The actual hardware, software versions, configuration, and prerequisites.
3. Reproduction steps and expected behavior.
4. Observed results with supporting logs or measurements.
5. Failures, troubleshooting steps, limitations, and lessons learned.
6. Resource cleanup and costs, where available.

Separate planned work from observed outcomes. Publish only reviewed, sanitized evidence; keep credentials, private data, datasets, checkpoints, and large generated artifacts out of Git.

This is an independent learning guide and portfolio.
