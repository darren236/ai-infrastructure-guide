# Progress Log

## Current state

- **Documentation:** [Guide 01](docs/guides/01-brev-gpu-node-validation.md) covers node validation, CUDA host/container architecture, Docker/NVIDIA integration, CUDA image inspection, and PyTorch CUDA access. [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) documents preflight and Nemotron model loading on GPU, with NSC train/dev downloads and pending single-file inference instructions for a small Singapore English ASR POC.
- **Guide 02:** Preflight, NeMo 3.0.0 container checks, model loading, persistent cache configuration, model placement on `cuda:0`, and NSC query/dev downloads and extraction complete. Tiny subsets, preprocessing, baseline inference, training, checkpointing, evaluation, and true streaming remain incomplete.
- **Hands-on guides:** Guide 01 in progress; cloud connectivity, Linux host, PCIe GPU visibility, and NVIDIA driver communication validated. Host Toolkit discovery complete for the checked PATH and conventional locations; no installation found.
- **Docker, NVIDIA container integration, container GPU passthrough, CUDA base image launch, and CUDA devel image / `nvcc` inspection:** Complete for the reported checks.
- **PyTorch NGC container, CUDA availability, GPU identity, and interactive/one-liner workflows:** Validated or documented for the reported scope; interactive execution reported.
- **Actual GPU inference/computation, real training, NeMo training, distributed training/NCCL, and performance benchmarking:** Not yet complete.
- **Training runs, application deployments, and benchmarks:** None documented.

## 2026-10-01 — Initial scaffold

Prepared the portfolio README, planned learning roadmap, progress log, Git ignore rules, and directories for documentation, guides, scripts, containers, Slurm, Kubernetes, benchmarks, and diagrams.

This entry records repository setup only. No cloud GPU provisioning or infrastructure experiments have been performed or validated as part of this setup.

**Planned next step at scaffold creation:** Identify a suitable cloud compute environment, capture sanitized hardware and runtime inspection output, and document the first environment inspection in `docs/guides/`.

## 2026-10-01 — Broader AI infrastructure scope

Broadened the portfolio framing to compute, orchestration, distributed systems, and operations. Updated the README and roadmap to describe general infrastructure concepts alongside tool-specific guides. Hands-on work remains planned.

## 2026-10-01 — Guide naming

Adopted the name **AI Infrastructure Guide** and updated the repository title, description, and README. The guide retains directories for practical guides and evidence as hands-on work is completed.

## 2026-10-01 — Guide 01: GPU Node Validation, Part 1 documented

Recorded the operator's sanitized milestone summary in [the guide](docs/guides/01-brev-gpu-node-validation.md). This entry's date records the documentation sync.

| Validation area | Status | Recorded evidence |
| --- | --- | --- |
| Brev provisioning and connectivity | Complete | CLI validated locally, authentication and organization selection completed, and `brev shell` succeeded after `brev refresh` and reauthentication. |
| Linux host identification | Complete | Execution context checked; Ubuntu 22.04.5 LTS, kernel `6.8.0-1069-gcp`, and x86_64 architecture identified. |
| PCIe GPU visibility | Complete | `lspci` enumerated an NVIDIA device. |
| NVIDIA driver communication | Complete | `nvidia-smi` detected an NVIDIA L4 with 23034 MiB memory and driver `595.91.07`. |
| CUDA userspace and Toolkit | Not yet complete | `nvidia-smi` reported compatibility level `13.2`; installed runtime/Toolkit versions and CUDA execution have not been validated. |
| Docker | Not yet complete | No validation reported. |
| NVIDIA Container Toolkit | Not yet complete | No validation reported. |
| PyTorch | Not yet complete | No validation reported. |
| NeMo | Not yet complete | No validation reported. |

**Observed state:** GPU idle, 0 MiB memory in use, 0% utilization, approximately 35°C, approximately 12 W against a 72 W cap, no running GPU processes, and no uncorrectable ECC errors reported. These observations describe one validation snapshot.

**Troubleshooting lesson:** Cloud VM running status did not establish working SSH access. Refreshing Brev connection configuration and reauthenticating restored connectivity; an underlying root cause was not independently isolated.

**Next step:** Validate CUDA userspace and Toolkit availability and versions, then distinguish them from driver compatibility. No raw terminal logs, authentication URLs, email addresses, organization identifiers, SSH configuration, tokens, or instance-specific connection details are included in the milestone.

## 2026-10-05 — Guide 01: CUDA host vs container architecture documented

Recorded the operator's latest sanitized milestone summary in [Guide 01, Part 2](docs/guides/01-brev-gpu-node-validation.md#part-2-cuda-host-vs-container-architecture). This entry's date records the documentation sync.

| Area | Status | Evidence or boundary |
| --- | --- | --- |
| Host driver and GPU access | Working | `nvidia-smi` succeeds and reports driver `595.91.07` and CUDA compatibility level `13.2`. |
| Host CUDA Toolkit discovery | Complete within the checked scope | `which nvcc` returned no path, `/usr/local/cuda/bin/nvcc` does not exist, and no conventional `/usr/local/cuda*` Toolkit installation was found. Alternate locations and environments were not exhaustively audited. |
| Host/container architecture | Documented | Added a conceptual diagram and explained the common pattern of a driver-enabled host with CUDA libraries, frameworks, and application dependencies supplied in containers. |
| CUDA runtime and workload execution | Not yet complete | Driver compatibility does not identify installed runtime/Toolkit versions or establish CUDA execution. |
| Docker | Not yet complete | Validation is a next step. |
| NVIDIA Container Toolkit | Not yet complete | GPU integration with the container runtime has not been validated. |
| Container CUDA Toolkit, PyTorch, and NeMo | Not yet complete | No container inspection or framework execution reported. |

**Learning point:** A working `nvidia-smi` with missing `nvcc` does not by itself indicate a broken GPU node. First identify the intended deployment model and where CUDA userspace should live. A host CUDA Toolkit is not required for the documented containerized pattern; this does not establish that Toolkit-free hosts are universal.

**Next step:** Validate Docker and NVIDIA Container Toolkit, then inspect and test CUDA GPU access inside the selected container. No raw logs or sensitive access details are included.

## 2026-10-05 — Guide 01: NVIDIA container and CUDA devel stack validated

Recorded the operator's sanitized milestone in [Guide 01, Part 3](docs/guides/01-brev-gpu-node-validation.md#part-3-docker-nvidia-container-gpu-path-and-cuda-devel-image). This date records the documentation sync; checks were performed by the operator.

| Area | Status | Reported evidence |
| --- | --- | --- |
| Docker installed and validated | Complete | `/usr/bin/docker`, Docker `29.8.2`, and successful container launches |
| NVIDIA container runtime/integration | Complete for tested path | NVIDIA CDI device names, registered `nvidia` runtime, default runtime `nvidia`, and successful container GPU query; package version not captured |
| GPU passthrough | Complete | `--gpus all` exposed the NVIDIA L4 inside the base container |
| CUDA base image | Complete for launch and GPU visibility | `nvidia/cuda:13.0.0-base-ubuntu22.04` ran `nvidia-smi`; `which nvcc` returned no path |
| CUDA devel image / `nvcc` | Complete for toolchain inspection | `nvidia/cuda:13.0.0-devel-ubuntu22.04` returned `/usr/local/cuda/bin/nvcc`, release `13.0`, `V13.0.48` |
| Actual CUDA kernel execution | Not yet complete | No program compiled or kernel executed |
| PyTorch CUDA validation | Not yet complete | No framework GPU test reported |
| NeMo workload | Not yet complete | No training run reported |

**Observations:** The uncached base image was pulled on first launch as expected. The container's `nvidia-smi` still displayed driver compatibility `13.2`; the devel compiler separately reported Toolkit `13.0`. No host Toolkit was found in the previously checked locations. The runtime image role is documented but that image was not tested. CDI configuration was observed without independently identifying the injection mode used for the GPU launch.

**SA lesson:** GPU visibility, development toolchain presence, CUDA kernel execution, framework GPU use, and real workloads are separate validation stages. A minimal CUDA program is the next lower-layer test before investigating framework behavior.

**Next step:** Compile and execute a tiny CUDA program in the GPU-enabled devel container, synchronize, and verify its result. Then validate PyTorch CUDA and NeMo. No raw logs or sensitive access details are included.

## 2026-10-05 — Guide 01: PyTorch CUDA access validated

Recorded the operator's sanitized milestone in [Guide 01, Part 4](docs/guides/01-brev-gpu-node-validation.md#part-4-pytorch-cuda-access-through-an-nvidia-ngc-container). This is the documentation sync date.

| Area | Status | Reported evidence |
| --- | --- | --- |
| PyTorch container | Complete for startup and inspection | `nvcr.io/nvidia/pytorch:26.09-py3`; NVIDIA Release `26.09`; PyTorch `2.14.0a0+b2c75dd062.nv26.09` |
| PyTorch CUDA availability | Complete | CUDA build `13.4`; `torch.cuda.is_available()` returned `True` |
| GPU identity from PyTorch | Complete | `torch.cuda.get_device_name(0)` returned `NVIDIA L4` |
| Interactive and one-liner workflows | Documented | Interactive checks performed; equivalent one-liner supplied for repeatable diagnostics, without claiming a separate run |
| Actual GPU tensor computation / standalone CUDA kernel test | Not yet complete | No computed result reported |
| Real training workload | Not yet complete | No training run reported |
| NeMo | Not yet complete | No NeMo workload reported |
| Distributed training / NCCL | Not yet complete | No collective or distributed run reported |
| Performance benchmarking | Not yet complete | No performance measurements reported |

**Compatibility observation:** The startup banner reported CUDA Forward Compatibility mode and CUDA 13.4 userspace compatibility components; the host kernel driver remained `595.91.07`. Container CUDA build and host compatibility reports are separate values. Successful availability/device queries do not establish support for every workload or feature.

**Troubleshooting lesson:** Misspelling `torch.cuda.is_available()` as `torch.cuda.is_availble()` caused an `AttributeError`; the correct call succeeded. Inspect user/application errors before escalating toward infrastructure.

**SA workflow:** Test the minimum needed to answer the customer's question in their real execution environment. Use host GPU query, container GPU query, PyTorch availability, real workload, and targeted deeper diagnostics as a troubleshooting ladder rather than a mandatory checklist.

**Next decision:** Keep Guide 01 in progress until its completion scope is agreed. A small GPU tensor computation with synchronization and verified output is the recommended final functional check before closing Guide 01. No raw logs or sensitive connection details are included.

## 2026-10-05 — Guide 02: node preflight documented

Started [Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](docs/guides/02-nemotron-streaming-singapore-english.md), intentionally scoped to a small POC/tutorial rather than production-quality tuning. This date records the documentation sync of the operator's sanitized summary.

Before pulling the NeMo container or downloading the model/data, ran `df -h`, `free -h`, and `nvidia-smi` on the GPU node.

**Observed resources:** Root filesystem 267 GB total, 53 GB used, and 214 GB available (operator-reported rounded units); host RAM 15 GiB total with approximately 14 GiB available; no swap. NVIDIA L4 with 23,034 MiB VRAM, 0 MiB used, 0% utilization, driver `595.91.07`, driver-reported CUDA compatibility `13.2`, and no running GPU processes.

**Learning point:** Filesystem capacity, system RAM, and GPU VRAM are separate resources. Linux filesystem cache can make `free` RAM low while `available` RAM remains high. Check these basics before investigating container/model failures; an idle resource snapshot does not establish capacity under workload.

| Guide 02 stage | Status |
| --- | --- |
| Node resource preflight | Complete |
| NeMo container validation | Not yet complete |
| Model loading | Not yet complete |
| Dataset preparation | Not yet complete |
| Training | Not yet complete |
| Evaluation | Not yet complete |

**Next step:** Validate the NeMo container for the small POC, then proceed to model loading and dataset preparation. No raw logs or sensitive connection details are included.

## 2026-10-05 — Guide 02: Nemotron model loaded on GPU

Updated [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete) with the operator's successful use of `nvcr.io/nvidia/nemo-speech:26.07.00` and `nvidia/nemotron-3.5-asr-streaming-0.6b`. Created `~/hf-cache` on the GPU host and bind-mounted it into `/root/.cache/huggingface` so cached files remain outside the disposable container.

**Observed result:** Loaded `EncDecRNNTBPEModelWithPrompt`; after `m.cuda()`, the parameter device printed `cuda:0`. This validates Docker GPU passthrough, the NeMo ASR import/model-loading path, and model placement on GPU 0. Earlier CUDA-container checks established the host-driver/container-userspace relationship. A second-run cache hit and actual transcription were not reported.

**Next milestone prepared:** Transcribe one small WAV in the same NeMo image, using persistent cache and a read-only host audio mount. Verified the NVIDIA-linked sample download (mono, 16 kHz, 7.4 seconds) and reviewed the documented transcription API. Instructions are ready for the SSH-accessed GPU node; inference and its resulting text remain pending.

Singapore English dataset preparation, adaptation/training, evaluation, true streaming, benchmarking, and scheduler submission remain incomplete. This guide stays a small POC/tutorial.

## 2026-10-05 — Guide 02: NeMo version and NSC train/dev data validated

Updated [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) with the operator's latest progress. Preflight confirmed 214 GB available disk, approximately 14 GiB available host RAM, and an idle NVIDIA L4 with 23,034 MiB VRAM and no GPU processes.

**Container/model:** `nvcr.io/nvidia/nemo-speech:26.07.00` reported NeMo `3.0.0`. Restored `nvidia/nemotron-3.5-asr-streaming-0.6b` as `EncDecRNNTBPEModelWithPrompt`; with the persistent host Hugging Face cache mounted, model parameters moved to `cuda:0`. The model itself fits on the L4 for loading; training capacity has not been established.

**Data:** Downloaded and extracted `nsc-query.tar.gz` and `nsc-dev.tar.gz` from `pengyizhou/IALP-2026-data`. Query data: `nsc_query_5h/manifest.jsonl` has 2,289 records; directory size reported as `214M`. Dev data: `nsc_dev_3h/manifest.jsonl` has 1,316 records. Inspected records share the source schema and relative FLAC audio paths; dev transcripts may contain `<v-noise>` annotation tags.

The supplied dev split is reserved for a cleaner before/after comparison, separate from fine-tuning data. Speaker/utterance disjointness still needs checking. Annotation tags will be removed during minimal preprocessing; no tag removal or NeMo manifest conversion has been performed.

**Next planned step:** Create deterministic POC subsets of approximately 300 training utterances and 50 dev utterances. Subset creation, manifest conversion, tag removal, baseline inference, training smoke test, fine-tuning, checkpointing, and post-training evaluation all remain incomplete. Existing one-WAV instructions remain an unexecuted reference. This remains a small POC/tutorial rather than production optimization.

## 2026-10-05 — Repository guide terminology aligned

Moved Guide 01 into `docs/guides/` alongside Guide 02 and renamed the empty code/configuration directory to `guides/`. Updated guide titles, terminology, local links, and the README directory tree throughout the repository. Aligned README and roadmap status with the latest Guide 02 milestone; completion boundaries and recorded results are unchanged.

## Updating this log

For each meaningful milestone, add the actual date, objective, work performed, links to evidence, observed result, blockers or lessons, and next step. Distinguish work in progress from completed and validated work.
