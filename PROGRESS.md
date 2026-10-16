# Learning Record

<a id="current-state"></a>

## Current guide outcomes

This page records the latest supported outcome for each guide and a concise milestone history. Detailed commands belong in the guides and scripts; superseded intermediate plans remain available in Git history rather than in the main reading path.

### Guide 01 — GPU Node Validation

[Guide 01](docs/guides/01-brev-gpu-node-validation.md) verifies the recorded path from an NVIDIA L4 and Linux visibility through the driver, container runtime, PyTorch/NeMo, and a real ASR application:

1. [Physical GPU](docs/guides/01-brev-gpu-node-validation.md#1-physical-gpu)
2. [PCIe enumeration](docs/guides/01-brev-gpu-node-validation.md#2-pcie-enumeration)
3. [Linux kernel](docs/guides/01-brev-gpu-node-validation.md#3-linux-kernel)
4. [NVIDIA driver](docs/guides/01-brev-gpu-node-validation.md#4-nvidia-driver)
5. [CUDA runtime / Toolkit](docs/guides/01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit)
6. [Container runtime](docs/guides/01-brev-gpu-node-validation.md#6-container-runtime)
7. [PyTorch / NeMo](docs/guides/01-brev-gpu-node-validation.md#7-pytorch--nemo)
8. [Application](docs/guides/01-brev-gpu-node-validation.md#8-application)

**Outcome:** The host-to-application path is verified for the reported L4 environment. A standalone CUDA micro-test, distributed execution, and performance measurement remain optional extensions rather than gaps in the core journey.

### Guide 02 — Nemotron ASR Adaptation

[Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) completes its nine-stage historical POC:

1. [Node resources](docs/guides/02-nemotron-streaming-singapore-english.md#1-node-resources)
2. [NeMo container](docs/guides/02-nemotron-streaming-singapore-english.md#2-nemo-container)
3. [Pretrained model](docs/guides/02-nemotron-streaming-singapore-english.md#3-pretrained-model)
4. [Data strategy and source checks](docs/guides/02-nemotron-streaming-singapore-english.md#4-data-strategy-and-source-checks)
5. [Training and validation inputs](docs/guides/02-nemotron-streaming-singapore-english.md#5-training-and-validation-inputs)
6. [Pretrained inference and baseline](docs/guides/02-nemotron-streaming-singapore-english.md#6-pretrained-inference-and-baseline)
7. [Training smoke test and artifacts](docs/guides/02-nemotron-streaming-singapore-english.md#7-training-smoke-test-and-artifacts)
8. [Checkpoint comparison and development](docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development)
9. [Held-out NSC test](docs/guides/02-nemotron-streaming-singapore-english.md#9-held-out-nsc-test)

| Evaluation | Pretrained | POC v1 | Difference |
| --- | ---: | ---: | ---: |
| Development: 50 recordings / 793 words | 10.844892812105927% | 10.466582597730138% | 86 → 83 errors |
| Held-out NSC: 3,670 recordings / 64,490 words | 12.566289347185611% | 12.254613118312916% | 201 fewer errors; 2.480257% relative reduction |

**Outcome:** POC v1 shows a modest offline in-domain improvement. Statistical significance, business-critical error quality, native streaming, latency, throughput, production capacity, and cost remain unvalidated.

**Historical limitations:** The 100-step run did not retain complete step-50 validation, sampling/exposure, loss/resource, or best/last/export provenance evidence. These limits constrain the claim but do not invalidate the frozen comparison.

#### Full-data extension — prepared, not executed

The separate [full-data workflow](scripts/guide-02/full-finetune/README.md) is ready for 2,152 train recordings / 4.50 hours and 1,203 development recordings / 2.65 hours. The [exclusions evidence](docs/evidence/guide-02-full-finetune-exclusions.csv) records decoder and duration removals.

Two runs capped at 10 GiB reached five steps and exited `137`; kernel evidence identified container memory-cgroup OOM kills. The next gate is a 32 GiB-or-larger node, duration/OOM profiling, and an independent 50-step resource pilot. No full-data candidate, streaming comparison, or new test result is claimed.

### Guide 03 — Scheduling GPU Workloads with Slurm

[Guide 03](docs/guides/03-scheduling-gpu-workloads-with-slurm.md) completes seven scheduler-use steps:

1. [Read cluster state](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#1-read-cluster-state)
2. [Read the configuration](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#2-read-the-configuration)
3. [Submit a CPU batch job](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#3-cpu-batch-job)
4. [Run interactively](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#4-interactive-srun)
5. [Submit a GPU job](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#5-gpu-allocation)
6. [Observe contention](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#6-gpu-contention)
7. [Schedule the existing workload](docs/guides/03-scheduling-gpu-workloads-with-slurm.md#7-scheduled-nemotron-workload)

**Outcome:** CPU batch, interactive CPU, explicit L4 allocation, single-GPU contention, and scheduled one-record Nemotron inference are verified for the reported environment. Cancellation/timeouts, persistent accounting, and multi-GPU-aware container binding remain optional extensions.

## Evidence boundaries

- Execution outcomes are operator-reported unless explicitly described as local repository checks.
- Documentation sync dates are not execution dates unless stated.
- A repository commit does not deploy scripts, data, or configuration to a GPU node.
- Raw data, audio, predictions, checkpoints, weights, caches, logs, credentials, and connection details remain outside Git.
- Functional results do not establish universal compatibility, sustained performance, statistical significance, or production readiness.

<a id="milestone-history"></a>

## Milestone history

### 2026-10-16 — Main reading path simplified

Reduced Guide 02 and this Learning Record to decisions, minimum useful checks, verified outcomes, and evidence boundaries. Removed repeated host bookkeeping, duplicate validation passes, abandoned alternatives, and superseded next-step prose from the main path. No script behavior or execution result changed; prior detail remains in Git history.

### 2026-10-16 — Full-data workflow prepared

Prepared 2,152 train and 1,203 development inputs, reviewed exclusions, and added the gated full-data workflow. Two 10 GiB-container attempts on the original node ended in confirmed memory-cgroup OOMs after five steps. The next evidence gate became a 32 GiB-or-larger-node profile and independent 50-step pilot.

### 2026-10-15 — Scheduled Nemotron inference completed

Verified the Guide 03 path through CPU batch, interactive work, explicit GPU allocation, contention, and a scheduled one-record Nemotron inference using the existing Guide 02 helper. The wrapper reached `Smoke test completed.` and its final timestamp.

### 2026-10-14 — POC v1 held-out comparison completed

Froze the 100-step pilot export and scoring policy before test inference. The finalized 3,670-record NSC test comparison changed WER from 12.566289347185611% to 12.254613118312916%, or 201 fewer errors.

### 2026-10-14 — Single-node Slurm environment established

Recorded a working Slurm `26.05.5` single-node environment with one L4, MUNGE, controller and worker services, cgroup v2 constraints, and a default `gpu` partition. Building upstream packages was an environment-specific exception, not part of the normal guide path.

### 2026-10-10 — Controlled pilot and development comparison completed

The 100-step pilot completed, saved/restored its export, and scored 10.466582597730138% WER on the fixed 50-record development set. The pretrained model scored 10.844892812105927%; the two-step smoke export scored 10.970996216897856%.

### 2026-10-09 — Inference, baseline, and official training path verified

Verified single-record pretrained inference, the 50-record / 793-word offline baseline, and a two-step official NeMo training run that saved checkpoints and a `.nemo` export on the L4.

### 2026-10-08 — POC manifests prepared

Normalized the declared transcript tags and converted the 300/50 deterministic subsets into Nemotron-compatible manifests while preserving source data and audio.

### 2026-10-07 — Data policy, audits, and subsets completed

Verified train/development speaker and utterance-ID separation, audited annotation tags, measured `<unk>` eligibility impact, declared the POC transcript policy, and generated the seed-42 speaker-aware subsets.

### 2026-10-05 — GPU, container, framework, and model path established

Recorded the Ubuntu/L4 environment, driver and container access, CUDA development container, PyTorch CUDA visibility, NeMo `3.0.0`, and pretrained Nemotron restoration on `cuda:0`.

### 2026-10-01 — Repository and first guide started

Created the shared-learning repository and began documenting the path from GPU inventory through the software stack. Later milestones supersede the intermediate pending states from the first entries.

## Updating this record

For a meaningful new milestone, record:

- the execution date and environment;
- the decision or question being answered;
- the minimum work performed;
- the observed result and evidence location;
- the boundary of the claim;
- the next gate only when further work is actually intended.

Do not append terminal diaries, repeated status checks, or a new entry for every documentation edit. Update the current outcome when new evidence changes it, and keep implementation detail in the relevant guide or script.
