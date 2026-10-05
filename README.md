# AI Infrastructure Guide

A hands-on guide to AI infrastructure and a technical portfolio covering compute, workload orchestration, distributed systems, and reliable operations. This repository will document reproducible labs, architecture decisions, operational lessons, and measured results as the work is completed.

The learning scope covers **GPU compute, containers, PyTorch/NeMo training, cloud infrastructure, Slurm, distributed communication, Kubernetes, observability, networking, storage, and troubleshooting**. Tool-specific labs will explore technologies such as CUDA, NCCL, and NVIDIA GPU Operator within that broader scope. The aim is to build practical experience designing and operating AI infrastructure and explain the tradeoffs clearly.

## Current status

**Lab 01, Part 1 complete:** [Brev GPU Node Validation](docs/labs/01-brev-gpu-node-validation.md) documents cloud connectivity, Linux host identification, PCIe GPU visibility, and NVIDIA driver communication on an L4 node.

**Latest milestone:** [CUDA host vs container architecture](docs/labs/01-brev-gpu-node-validation.md#part-2-cuda-host-vs-container-architecture) documents working host driver access, no Toolkit found in PATH or conventional locations, and the intended containerized deployment pattern.

CUDA runtime execution, container CUDA Toolkit inspection, Docker, NVIDIA Container Toolkit, PyTorch, and NeMo validation remain pending. No training runs, deployments, benchmarks, or complete end-to-end lab are claimed.

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
│   └── labs/         # Step-by-step lab write-ups and evidence
├── labs/             # Lab-specific code and configuration
├── scripts/          # Reusable setup, inspection, and validation scripts
├── containers/       # Container definitions and workload environments
├── slurm/            # Scheduler configuration and job scripts
├── kubernetes/       # Manifests and GPU platform configuration
├── benchmarks/       # Benchmark methods and verified measurements
└── diagrams/         # Architecture diagrams and editable sources
```

Empty directories contain `.gitkeep` placeholders so Git preserves the scaffold.

## Continue the guide

Start with [Lab 01: GPU Node Validation](docs/labs/01-brev-gpu-node-validation.md) for the environment inventory, connectivity recovery, host Toolkit discovery, and host/container architecture. Next, validate Docker and NVIDIA Container Toolkit, then inspect and test CUDA inside the selected container.

## Documentation standard

Each completed lab should record:

1. The objective and architecture, including relevant tradeoffs.
2. The actual hardware, software versions, configuration, and prerequisites.
3. Reproduction steps and expected behavior.
4. Observed results with supporting logs or measurements.
5. Failures, troubleshooting steps, limitations, and lessons learned.
6. Resource cleanup and costs, where available.

Separate planned work from observed outcomes. Publish only reviewed, sanitized evidence; keep credentials, private data, datasets, checkpoints, and large generated artifacts out of Git.

This is an independent learning guide and portfolio.
