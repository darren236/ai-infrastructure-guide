# NVIDIA Solutions Architect Lab

A hands-on NVIDIA AI infrastructure learning journey and technical portfolio. This repository will document reproducible labs, architecture decisions, operational lessons, and measured results as the work is completed.

The learning scope covers **CUDA, containers, NeMo/PyTorch training, cloud GPUs, Slurm, NCCL, Kubernetes, NVIDIA GPU Operator, observability, networking, storage, and troubleshooting**. The aim is to build practical experience connecting these components into reliable AI infrastructure and explain the tradeoffs clearly.

## Current status

**Initial documentation scaffold.** Hands-on labs have not started. No training runs, deployments, benchmarks, or completed labs are claimed.

Follow the planned milestones in [ROADMAP.md](ROADMAP.md) and the evidence-backed activity log in [PROGRESS.md](PROGRESS.md).

## Learning objectives

- Understand the GPU software stack, CUDA compatibility, resource usage, and failure modes.
- Run reproducible containerized NeMo/PyTorch workloads on cloud GPUs.
- Schedule GPU jobs with Slurm and investigate distributed communication with NCCL.
- Operate GPU workloads on Kubernetes using NVIDIA GPU Operator.
- Evaluate observability, networking, and storage requirements, and document troubleshooting decisions.

## Repository layout

```text
nvidia-sa-lab/
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

## Starting point

The first planned lab is to identify a cloud GPU and inspect its driver, available memory, and reported CUDA compatibility using `nvidia-smi`. Provisioning details, commands, and observations will be recorded after the lab is performed.

## Documentation standard

Each completed lab should record:

1. The objective and architecture, including relevant tradeoffs.
2. The actual hardware, software versions, configuration, and prerequisites.
3. Reproduction steps and expected behavior.
4. Observed results with supporting logs or measurements.
5. Failures, troubleshooting steps, limitations, and lessons learned.
6. Resource cleanup and costs, where available.

Separate planned work from observed outcomes. Publish only reviewed, sanitized evidence; keep credentials, private data, datasets, checkpoints, and large generated artifacts out of Git.

This is an independent learning portfolio and does not imply NVIDIA sponsorship or certification.
