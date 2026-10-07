# Progress Log

## Current state

- **Documentation:** [Guide 01](docs/guides/01-brev-gpu-node-validation.md) covers node validation, CUDA host/container architecture, Docker/NVIDIA integration, CUDA image inspection, and PyTorch CUDA access. [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) documents preflight, model loading, NSC source downloads, and a four-stage experiment overview for a small Singapore English ASR POC. Single-file inference instructions remain unexecuted.
- **Guide 02:** Preflight, NeMo 3.0.0 container checks, model loading/cache/GPU placement, source inspection, train/validation speaker/utterance-ID separation, tag enumeration, eligibility analysis, the POC policy, deterministic speaker-aware subset generation, and independent subset checks complete. `train_300.jsonl` and `dev_50.jsonl` were created and checked for ownership, record counts, tags, speakers, and separation. Referenced-audio integrity, normalization output, NeMo conversion, inference, training, checkpointing, evaluation, and true streaming remain incomplete.
- **Guide 02 data roles:** `nsc_query_5h` is train and `nsc_dev_3h` is validation (dev), both from the NSC Part 6 train partition. Local checks found 111/64 speakers, zero shared speakers, and zero shared utterance IDs. [Train → validation → freeze model/configuration → NSC test → GigaSpeech](docs/guides/02-nemotron-streaming-singapore-english.md#data-strategy-and-experiment-overview) remains planned execution; test/benchmark preparation and separation from `nsc_test` remain unvalidated.
- **Guide 02 annotations:** Source audits found `<v-noise>`, `<unk>`, and `<noise>`; 511 train and 286 validation records contain tags. Derived checks confirmed `<unk>` is absent. Train: 61 tagged records, `<v-noise>` 87 / `<noise>` 5; validation: 4 tagged records, `<v-noise>` 7. Noise-tag removal and transcript normalization remain pending, with local speech retained. Manual listening/relabeling is excluded; source manifests remain unchanged.
- **Guide 02 subsets:** Eligible pools were 2,158 train / 4.59 h and 1,225 validation / 2.71 h. Seed-42 generation reported 300 train / 111 speakers / 0.66 h and 50 validation / 50 speakers / 0.11 h. Independent host/helper checks confirmed record/speaker counts, zero speaker/ID overlap, and `<unk>` absence. The two derived files were corrected from `root:root` to `ubuntu:ubuntu`; original NSC files were unchanged.
- **Hands-on guides:** Guide 01 in progress; cloud connectivity, Linux host, PCIe GPU visibility, and NVIDIA driver communication validated. Host Toolkit discovery complete for the checked PATH and conventional locations; no installation found.
- **Docker, NVIDIA container integration, container GPU passthrough, CUDA base image launch, and CUDA devel image / `nvcc` inspection:** Complete for the reported checks.
- **PyTorch NGC container, CUDA availability, GPU identity, and interactive/one-liner workflows:** Validated or documented for the reported scope; interactive execution reported.
- **Actual GPU inference/computation, real training, NeMo training, distributed training/NCCL, and performance benchmarking:** Not yet complete.
- **Training runs, application deployments, and benchmarks:** None documented.

### Guide 02 end-of-day checkpoint

**Current checkpoint:** POC train/validation subsets have been created and independently validated. Train: 300 records / 111 speakers / ~0.66 h; validation: 50 / 50 / ~0.11 h. Both have zero `<unk>`, with zero shared speakers or IDs. POC subset preparation is complete; normalization and NeMo conversion are not.

**Pipeline:** The [checked overview](docs/guides/02-nemotron-streaming-singapore-english.md#current-pipeline-status) places transcript normalization immediately after derived-subset validation. Baseline inference, baseline validation WER, training smoke test, fine-tuning, validation/model selection, and checkpoint/configuration freeze remain pending. `nsc_test` and `gigaspeech_test` have no documented local download or evaluation; both evaluations are not started.

**Next session:** Define and implement normalization for `<v-noise>`/`<noise>` while preserving spoken words, identifiable Singapore-English/Singlish speech, and fillers. Create new derived artifacts separately or reproducibly during NeMo conversion; original manifests remain unchanged. Then convert to NeMo ASR manifests.

**Reproducibility scope:** Seed 42 and deterministic speaker-aware sampling are already recorded. SHA-256 hashing/byte-for-byte regeneration is optional additional rigor for stricter production pipelines, deliberately not a required tutorial blocker.

## Milestone history

The dated entries below record status and next steps at each milestone; earlier pending items may have been completed in later entries. Use **Current state** above and the guide completion sections for the latest status. Entry dates record documentation sync unless an execution date is explicitly stated. Historical Part 1–4 names refer to Guide 01's earlier organization; it now follows the eight-layer agenda.

### 2026-10-01 — Initial scaffold

Prepared the portfolio README, planned learning roadmap, progress log, Git ignore rules, and directories for documentation, guides, scripts, containers, Slurm, Kubernetes, benchmarks, and diagrams.

This entry records repository setup only. No cloud GPU provisioning or infrastructure experiments have been performed or validated as part of this setup.

**Planned next step at scaffold creation:** Identify a suitable cloud compute environment, capture sanitized hardware and runtime inspection output, and document the first environment inspection in `docs/guides/`.

### 2026-10-01 — Broader AI infrastructure scope

Broadened the portfolio framing to compute, orchestration, distributed systems, and operations. Updated the README and roadmap to describe general infrastructure concepts alongside tool-specific guides. Hands-on work remains planned.

### 2026-10-01 — Guide naming

Adopted the name **AI Infrastructure Guide** and updated the repository title, description, and README. The guide retains directories for practical guides and evidence as hands-on work is completed.

### 2026-10-01 — Guide 01: GPU Node Validation, Part 1 documented

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

### 2026-10-05 — Guide 01: CUDA host vs container architecture documented

Recorded the operator's latest sanitized milestone summary in [Guide 01, layer 5](docs/guides/01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit). This entry's date records the documentation sync.

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

### 2026-10-05 — Guide 01: NVIDIA container and CUDA devel stack validated

Recorded the operator's sanitized milestone in [Guide 01, layer 6](docs/guides/01-brev-gpu-node-validation.md#6-container-runtime). This date records the documentation sync; checks were performed by the operator.

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

### 2026-10-05 — Guide 01: PyTorch CUDA access validated

Recorded the operator's sanitized milestone in [Guide 01, layer 7](docs/guides/01-brev-gpu-node-validation.md#7-pytorch--nemo). This is the documentation sync date.

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

### 2026-10-05 — Guide 02: node preflight documented

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

### 2026-10-05 — Guide 02: Nemotron model loaded on GPU

Updated [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete) with the operator's successful use of `nvcr.io/nvidia/nemo-speech:26.07.00` and `nvidia/nemotron-3.5-asr-streaming-0.6b`. Created `~/hf-cache` on the GPU host and bind-mounted it into `/root/.cache/huggingface` so cached files remain outside the disposable container.

**Observed result:** Loaded `EncDecRNNTBPEModelWithPrompt`; after `m.cuda()`, the parameter device printed `cuda:0`. This validates Docker GPU passthrough, the NeMo ASR import/model-loading path, and model placement on GPU 0. Earlier CUDA-container checks established the host-driver/container-userspace relationship. A second-run cache hit and actual transcription were not reported.

**Next milestone prepared:** Transcribe one small WAV in the same NeMo image, using persistent cache and a read-only host audio mount. Verified the NVIDIA-linked sample download (mono, 16 kHz, 7.4 seconds) and reviewed the documented transcription API. Instructions are ready for the SSH-accessed GPU node; inference and its resulting text remain pending.

Singapore English dataset preparation, adaptation/training, evaluation, true streaming, benchmarking, and scheduler submission remain incomplete. This guide stays a small POC/tutorial.

### 2026-10-05 — Guide 02: NeMo version and NSC train/dev data validated

Updated [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md) with the operator's latest progress. Preflight confirmed 214 GB available disk, approximately 14 GiB available host RAM, and an idle NVIDIA L4 with 23,034 MiB VRAM and no GPU processes.

**Container/model:** `nvcr.io/nvidia/nemo-speech:26.07.00` reported NeMo `3.0.0`. Restored `nvidia/nemotron-3.5-asr-streaming-0.6b` as `EncDecRNNTBPEModelWithPrompt`; with the persistent host Hugging Face cache mounted, model parameters moved to `cuda:0`. The model itself fits on the L4 for loading; training capacity has not been established.

**Data:** Downloaded and extracted `nsc-query.tar.gz` and `nsc-dev.tar.gz` from `pengyizhou/IALP-2026-data`. Query data: `nsc_query_5h/manifest.jsonl` has 2,289 records; directory size reported as `214M`. Dev data: `nsc_dev_3h/manifest.jsonl` has 1,316 records. Inspected records share the source schema and relative FLAC audio paths; dev transcripts may contain `<v-noise>` annotation tags.

The supplied dev split is reserved for a cleaner before/after comparison, separate from fine-tuning data. Speaker/utterance disjointness still needs checking. Annotation tags will be removed during minimal preprocessing; no tag removal or NeMo manifest conversion has been performed.

**Next planned step:** Create deterministic POC subsets of approximately 300 training utterances and 50 dev utterances. Subset creation, manifest conversion, tag removal, baseline inference, training smoke test, fine-tuning, checkpointing, and post-training evaluation all remain incomplete. Existing one-WAV instructions remain an unexecuted reference. This remains a small POC/tutorial rather than production optimization.

### 2026-10-05 — Repository guide terminology aligned

Moved Guide 01 into `docs/guides/` alongside Guide 02 and renamed the empty code/configuration directory to `guides/`. Updated guide titles, terminology, local links, and the README directory tree throughout the repository. Aligned README and roadmap status with the latest Guide 02 milestone; completion boundaries and recorded results are unchanged.

### 2026-10-06 — Repository navigation organized

Added a guide index and navigation links in both guides, simplified the repository overview, and documented where future runnable guide assets should live. Removed empty placeholder directories; planned technologies remain in the roadmap and will receive directories when actual files are added. Recorded results and guide completion boundaries are unchanged.

### 2026-10-07 — Eight-layer validation agenda added

Organized Guide 01 around physical GPU, PCIe enumeration, Linux kernel, NVIDIA driver, CUDA runtime/Toolkit, container runtime, PyTorch/NeMo, and application. Added the same linked agenda to the overview/index and positioned Guide 02 as the application continuation. Preserved historical section anchors for existing milestone links.

Clarified that `nvidia-smi` reports driver-supported CUDA compatibility, while `nvcc` identifies the development Toolkit in the selected environment. A host Toolkit is not required for this containerized setup. NeMo import/model placement evidence is linked from Guide 02; actual GPU computation, ASR inference, training, and evaluation remain pending. This is a documentation reorganization, with no new remote validation claimed.

### 2026-10-07 — Repository consistency reviewed

Reviewed all repository files against the eight-layer agenda and reported guide evidence. Separated current status from historical milestone snapshots, aligned framework/cache wording, and updated historical links to the current layer sections. Clarified the host/container diagram and kept the SA troubleshooting checks distinct from agenda numbering.

Documented a host data directory outside the checkout and added ignore rules for Guide 02 cache, audio, dataset directories, and archives. Dataset split separation remains a publisher-described property awaiting local overlap checks. The next step remains approximately 300 training / 50 dev utterances; no new GPU execution, inference, preprocessing, or training is claimed.

### 2026-10-07 — Guide 02 data strategy and experiment overview documented

Added the four data roles near the beginning of [Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md#data-strategy-and-experiment-overview), with a sequential diagram, source/size tables, allowed learning and interpretation boundaries, and a 12-step workflow agenda. Training updates weights directly; validation influences model/configuration choices indirectly without backpropagation. Final test use follows finalized development decisions; repeatedly tuning on test results turns that set into validation.

Recorded the known NSC source counts and approximate sizes while keeping ~300/~50 subsets uncreated. Speaker/utterance-ID verification, normalization, NeMo manifests, baseline inference, training, checkpointing, and evaluation remain pending. Final test selection and optional external benchmark selection are explicit later TODOs. Added reproducibility and remote-host data-path guidance; no new datasets or GPU execution are claimed.

### 2026-10-07 — Repository organized around per-guide agendas

Made the repository README the starting point, with a separate goal, full agenda, and status for each active guide: eight stack layers for Guide 01 and the 12-step ASR workflow for Guide 02. Reduced the directory index to guide links and reserved the roadmap for future topics.

Removed the unused top-level `guides/` placeholder; written guides remain in `docs/guides/` at their existing paths. Moved Guide 02's agenda ahead of its detailed data strategy and placed the optional single-WAV smoke test in an appendix after the main progress section. Existing commands, results, and completion boundaries are preserved; ~300/50 subsets remain the next execution step.

### 2026-10-07 — Guide 02 NSC test and GigaSpeech OOD plan confirmed

Checked the [upstream dataset documentation](https://huggingface.co/datasets/pengyizhou/IALP-2026-data) and updated Guide 02/README agendas: query (~300 planned) → dev (~50 planned) → freeze configuration → `nsc_test` (3,684 utterances / ~7 h upstream) → `gigaspeech_test` (19,930 / 35.4 h upstream). Local query/dev counts remain 2,289/1,316; the POC subsets are not created.

Documented that query/dev come from different speaker sets in the NSC train partition and are not subsets of one another. Upstream states query/dev/test are mutually speaker-disjoint and query/dev exclude all official-test speakers. Pairwise speaker/utterance-ID checks on the compute node remain a hands-on reproducibility step. Test/benchmark preparation, evaluation, and all pending training work remain unvalidated; no new downloads or GPU execution are claimed.

### 2026-10-07 — Guide 02 annotations and data-preparation sequence documented

Recorded the operator's finding that `<v-noise>` occurs in both training/query and validation/dev transcripts as a vocal/non-lexical noise annotation. A full annotation inventory and counts remain pending; no final removal or replacement rule is selected. This supersedes the earlier tag-removal plan.

Aligned Guide 02 and the README agenda: source checks → speaker/utterance-ID separation → annotation inspection/counts → deterministic ~300/~50 subsets → normalization policy → derived NeMo manifests → baseline validation → train → validation/model selection → freeze → `nsc_test` → `gigaspeech_test`. Original manifests remain untouched. All preparation, inference, training, and evaluation steps in this sequence are planned, with no new node execution claimed.

### 2026-10-07 — Guide 02 train/validation separation and annotation audit verified

Recorded the operator's read-only Brev-node checks of `/data/nsc/nsc_query_5h/manifest.jsonl` and `/data/nsc/nsc_dev_3h/manifest.jsonl`: 2,289/1,316 records, 111/64 speakers, zero speaker overlap, and zero utterance-ID overlap. This independently confirms train/dev separation, without claiming local NSC test verification.

The complete-manifest annotation audit found 511/286 tagged records. Train/validation token occurrences: `<v-noise>` 598/345, `<unk>` 145/104, and `<noise>` 49/24. Checks used reusable scripts in the pinned NeMo Speech container with source data mounted read-only, without installing host Python or modifying the dataset.

Documented the intended annotation meanings and the unverified possibility that some `<unk>` labels could hide identifiable local speech. No audio listening has confirmed that hypothesis. Manual listening/relabeling is outside this reproducible POC; a reviewed derived dataset is future work. No final normalization/filtering rule is selected. Subsets, NeMo conversion, inference, and training remain unperformed.

### 2026-10-07 — Guide 02 validated audit helpers added

Added the supplied, node-validated [`check_split_overlap.py`](scripts/guide-02/check_split_overlap.py) and [`inspect_transcript_tags.py`](scripts/guide-02/inspect_transcript_tags.py) unchanged under `scripts/guide-02/`. Guide 02 links to the helpers, documents runtime manifest arguments and read-only NeMo-container reproduction commands, and preserves the verified Brev results.

This publishes existing validation tools and evidence; no new node run is claimed. Subset creation, normalization, NeMo conversion, inference, and training remain pending. `check_poc_eligibility.py` is not added because it has not been validated on the node.

### 2026-10-07 — Guide 02 POC eligibility analysis and annotation policy validated

Added the supplied [`check_poc_eligibility.py`](scripts/guide-02/check_poc_eligibility.py) unchanged. Recorded the exact Brev-host command: the pinned NeMo container mounted `/home/ubuntu/data/nsc` and `/home/ubuntu/work/nemotron-poc` read-only as `/data/nsc` and `/work`. Manifest paths are runtime arguments; this CPU/data-integrity task needs no GPU access.

Observed train: 2,289 records / 5.01 h, with 131 `<unk>` records / 0.42 h; 2,158 / 4.59 h would remain. Validation: 1,316 / 3.02 h, with 91 `<unk>` records / 0.31 h; 1,225 / 2.71 h would remain. Hours sum manifest durations; the read-only check creates no filtered output.

Marked overlap verification, annotation enumeration, eligibility analysis, and the POC handling decision complete. The policy excludes whole `<unk>` utterances, keeps `<noise>`/`<v-noise>` utterances for later token removal, retains local speech, and preserves originals. This is a pragmatic POC choice, not a bad-data judgment; manual audio review remains future work. Subsets, normalized output, NeMo conversion, baseline inference, training, checkpointing, and test/external evaluation remain pending.

### 2026-10-07 — Guide 02 deterministic speaker-aware subsets generated

Added the supplied, node-run [`create_poc_subsets.py`](scripts/guide-02/create_poc_subsets.py) unchanged. After excluding `<unk>`, the eligible pools were 2,158 train / 4.59 h and 1,225 validation / 2.71 h. The POC chooses 300/50 for fast NVIDIA L4 validation, with speaker cycling for train, one utterance per validation speaker, and separate RNG streams (`seed = 42`, validation `seed + 1`).

The operator created `~/data/nsc/poc` and successfully ran the pinned NeMo container with source/tooling mounted read-only and the dedicated derived directory writable at `/output`. The generator reported `train_300.jsonl`: 300 records, 111 speakers, 0.66 h; `dev_50.jsonl`: 50 records, 50 speakers, 0.11 h. Source manifests remain unchanged; generated records are not yet normalized or converted to NeMo format.

Recorded the arbitrary-host-UID failures (`python3: not found`, then `/opt/venv/bin/python3: Permission denied`). The successful command uses the image's default context; ownership inspection and any needed correction remain pending.

Subset generation and file creation are complete, while independent ownership, line-count, `<unk>` absence, speaker-count, and reproduction checks remain TODOs. Normalization, NeMo conversion, baseline inference, fine-tuning, checkpointing, validation WER, NSC test, and GigaSpeech evaluation remain pending.

### 2026-10-07 — Guide 02 derived POC manifests independently validated

Recorded the operator's completed [subset checks](docs/guides/02-nemotron-streaming-singapore-english.md#derived-poc-subset-validation--complete). The source manifests under `/home/ubuntu/data/nsc/nsc_query_5h` and `/home/ubuntu/data/nsc/nsc_dev_3h` remain unchanged. The validated subset generator and audit helpers are retained without code changes.

**Host checks:** `ls -lh` found root-owned derived files (82K train / 13K dev). Ownership was corrected only for `poc/train_300.jsonl` and `poc/dev_50.jsonl` to `ubuntu:ubuntu`. Independent `wc -l` returned 300 / 50 records.

**Derived audits:** Train has 61 tagged records, `<v-noise>` 87 and `<noise>` 5; validation has 4 tagged records, `<v-noise>` 7. Both contain zero `<unk>`. The overlap helper confirmed 111 train / 50 validation speakers, zero shared speakers, and zero shared IDs. Read-only data/tooling mounts in the pinned NeMo container kept the checks separate from source modification or GPU workload validation.

**Operational lessons:** The earlier arbitrary-UID failure was handled with the image's default execution context, protected mounts, and explicit derived-file ownership correction. Unexpected tag output from the node's overwritten overlap script prompted artifact inspection, local preservation as `check_split_overlap.py.bad`, restoration of the known-good helper, and rerunning validation. The incorrect copy is not committed.

**Next:** Normalize only the noise annotation tokens in separate derived outputs, preserving actual words and local speech; then convert NeMo manifests. Normalization, conversion, baseline inference, training, checkpointing, WER evaluation, NSC test, and GigaSpeech evaluation remain pending. Byte-for-byte reproduction testing is not a required tutorial gate. Manual audio review/relabeling remains future work outside this POC.

### 2026-10-07 — Guide 02 end-of-day checkpoint synchronized

Aligned the guide's pipeline, four-stage status table, README handoff, and current-state log with the existing validation evidence. Source-format train/validation subset preparation is complete; normalization is the exact next task, followed by NeMo conversion, baseline inference/WER, and training. NSC test and GigaSpeech evaluation remain unstarted, without documented local downloads.

Recorded a clear stopping point and next-session handoff. Seed 42/deterministic sampling remains the POC reproducibility design; hashing or byte-for-byte regeneration is optional production rigor. Existing ownership, count, tag, and separation results are retained. This is a status/documentation update, with no new scripts, normalized artifacts, dataset downloads, or GPU execution claimed.

## Updating this log

For each meaningful milestone, add a dated entry with the objective, work performed, links to evidence, observed result, blockers or lessons, and next step. Identify documentation sync dates and execution dates separately when known. Update **Current state** to match the latest evidence, while preserving earlier milestone snapshots. Distinguish prepared instructions from performed and validated work.
