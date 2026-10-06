# Guide index

[Repository overview](../../README.md) · [Progress](../../PROGRESS.md) · [Roadmap](../../ROADMAP.md)

## 01 — GPU Node Validation

[Open Guide 01](01-brev-gpu-node-validation.md)

Validate the infrastructure from host access through driver, container, and framework checks on an NVIDIA L4.

- [Brev connectivity, Linux host, PCIe, and driver](01-brev-gpu-node-validation.md#part-1-brev-connectivity-linux-host-pcie-and-nvidia-driver)
- [CUDA host vs container architecture](01-brev-gpu-node-validation.md#part-2-cuda-host-vs-container-architecture)
- [Docker, container GPU access, and CUDA development tools](01-brev-gpu-node-validation.md#part-3-docker-nvidia-container-gpu-path-and-cuda-devel-image)
- [PyTorch CUDA access](01-brev-gpu-node-validation.md#part-4-pytorch-cuda-access-through-an-nvidia-ngc-container)

**Status:** Reported access and inspection checks are complete. A real GPU computation remains pending; the guide is in progress.

## 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](02-nemotron-streaming-singapore-english.md)

Build a small POC/tutorial using the NeMo Speech container, a persistent model cache, and separate NSC query/dev data.

**Completed:** Node preflight, NeMo 3.0.0 checks, model loading on `cuda:0`, cache configuration, and dataset downloads/extraction.

**Next:** Deterministic subsets of approximately 300 training utterances and 50 dev utterances. Manifest conversion, tag cleanup, baseline inference, training, checkpointing, and evaluation remain pending. The guide includes a one-WAV inference reference that has not been executed on the GPU node.

## Future guides

Storage, Slurm, distributed training/NCCL, Kubernetes/GPU Operator, observability, and architecture synthesis remain planned in the [roadmap](../../ROADMAP.md). New guides are added here when work begins.
