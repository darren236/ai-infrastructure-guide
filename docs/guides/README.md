# Guide index

The guides form one sequential learning journey. Start with the [overview and shared agenda](../../README.md#guides-and-agendas), then use each guide for detailed commands, evidence, and boundaries.

## [Guide 01 — GPU Node Validation](01-brev-gpu-node-validation.md)

Which layer should answer a GPU troubleshooting question?

1. [Physical GPU](01-brev-gpu-node-validation.md#1-physical-gpu)
2. [PCIe enumeration](01-brev-gpu-node-validation.md#2-pcie-enumeration)
3. [Linux kernel](01-brev-gpu-node-validation.md#3-linux-kernel)
4. [NVIDIA driver](01-brev-gpu-node-validation.md#4-nvidia-driver)
5. [CUDA runtime / Toolkit](01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit)
6. [Container runtime](01-brev-gpu-node-validation.md#6-container-runtime)
7. [PyTorch / NeMo](01-brev-gpu-node-validation.md#7-pytorch--nemo)
8. [Application](01-brev-gpu-node-validation.md#8-application)

**What we learned:** The recorded L4 path is verified from host to application; each check proves only its stated layer and scope.

## [Guide 02 — Nemotron ASR Adaptation](02-nemotron-streaming-singapore-english.md)

How can model adaptation keep training, development selection, and held-out testing distinct?

1. [Node resources](02-nemotron-streaming-singapore-english.md#1-node-resources)
2. [NeMo container](02-nemotron-streaming-singapore-english.md#2-nemo-container)
3. [Pretrained model](02-nemotron-streaming-singapore-english.md#3-pretrained-model)
4. [Data strategy and source checks](02-nemotron-streaming-singapore-english.md#4-data-strategy-and-source-checks)
5. [Training and validation inputs](02-nemotron-streaming-singapore-english.md#5-training-and-validation-inputs)
6. [Pretrained inference and baseline](02-nemotron-streaming-singapore-english.md#6-pretrained-inference-and-baseline)
7. [Training smoke test and artifacts](02-nemotron-streaming-singapore-english.md#7-training-smoke-test-and-artifacts)
8. [Checkpoint comparison and development](02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development)
9. [Held-out NSC test](02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test)

**What we learned:** POC v1 reduced WER from 12.5663% to 12.2546% over 3,670 recordings / 64,490 words, a modest offline in-domain result.

## [Guide 03 — Scheduling GPU Workloads with Slurm](03-scheduling-gpu-workloads-with-slurm.md)

How does Slurm allocate CPU/GPU resources and launch an existing workload?

1. [Read cluster state](03-scheduling-gpu-workloads-with-slurm.md#1-read-cluster-state)
2. [Read the configuration](03-scheduling-gpu-workloads-with-slurm.md#2-read-the-configuration)
3. [Submit a CPU batch job](03-scheduling-gpu-workloads-with-slurm.md#3-cpu-batch-job)
4. [Run interactively](03-scheduling-gpu-workloads-with-slurm.md#4-interactive-srun)
5. [Submit a GPU job](03-scheduling-gpu-workloads-with-slurm.md#5-gpu-allocation)
6. [Observe contention](03-scheduling-gpu-workloads-with-slurm.md#6-gpu-contention)
7. [Schedule the existing workload](03-scheduling-gpu-workloads-with-slurm.md#7-scheduled-nemotron-workload)

**What we learned:** The single-node path works through scheduled Nemotron inference; allocation, utilization, accounting, and production operations remain separate concerns.

Each core guide is complete for its stated learning outcome. Optional extensions—including a CUDA micro-test, full-data ASR training, native streaming comparison, GigaSpeech evaluation, lifecycle controls, accounting, and multi-GPU-aware container integration—are collected in [Further Learning](../../ROADMAP.md).

Guide 02 also provides a [section map](02-nemotron-streaming-singapore-english.md#guide-sections), [nine-stage agenda](02-nemotron-streaming-singapore-english.md#hands-on-agenda), [tooling index](../../scripts/guide-02/README.md), and [optional full-data runbook](../../scripts/guide-02/full-finetune/README.md).

See the [Learning Record](../../PROGRESS.md) for current outcomes and dated evidence.
