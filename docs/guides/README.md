# Guide index

[Repository overview](../../README.md) · [Progress](../../PROGRESS.md) · [Roadmap](../../ROADMAP.md)

## 01 — GPU Node Validation

[Open Guide 01](01-brev-gpu-node-validation.md)

Validate the infrastructure from host access through driver, container, and framework checks on an NVIDIA L4.

Read the guide down the eight-layer agenda:

1. [Physical GPU](01-brev-gpu-node-validation.md#1-physical-gpu)
2. [PCIe enumeration — `lspci`](01-brev-gpu-node-validation.md#2-pcie-enumeration)
3. [Linux kernel](01-brev-gpu-node-validation.md#3-linux-kernel)
4. [NVIDIA driver — `nvidia-smi`](01-brev-gpu-node-validation.md#4-nvidia-driver)
5. [CUDA runtime / Toolkit — host versus container](01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit)
6. [Container runtime — Docker](01-brev-gpu-node-validation.md#6-container-runtime)
7. [PyTorch / NeMo — framework checks](01-brev-gpu-node-validation.md#7-pytorch--nemo)
8. [Application — continue with Guide 02](01-brev-gpu-node-validation.md#8-application)

**Status:** Reported access and inspection checks are complete. A real GPU computation remains pending; the guide is in progress.

## 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](02-nemotron-streaming-singapore-english.md)

This continues the agenda at **layer 8: application**, following NeMo framework/model-loading checks in layer 7. Build a small POC/tutorial using the NeMo Speech container, a persistent model cache, and separate NSC query/dev data.

**Completed:** Node preflight, NeMo 3.0.0 checks, model loading on `cuda:0`, cache configuration, and dataset downloads/extraction.

**Next:** Deterministic subsets of approximately 300 training utterances and 50 dev utterances. Manifest conversion, tag cleanup, baseline inference, training, checkpointing, and evaluation remain pending. The guide includes a one-WAV inference reference that has not been executed on the GPU node.

## Future guides

Storage, Slurm, distributed training/NCCL, Kubernetes/GPU Operator, observability, and architecture synthesis remain planned in the [roadmap](../../ROADMAP.md). New guides are added here when work begins.
