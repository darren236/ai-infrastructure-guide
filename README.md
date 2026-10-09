# AI Infrastructure Guide

Practical AI infrastructure guides and a technical portfolio demonstrating NVIDIA Solutions Architect skills: GPU stack validation, containerized workloads, troubleshooting, and clear explanations of architecture and operational tradeoffs.

Start with **Guide 01** to understand and validate the GPU stack, then follow **Guide 02** into a small Nemotron ASR fine-tuning POC. Each guide's agenda is below; open the guide for commands, explanations, observed results, and pending work.

Use the [current progress](PROGRESS.md#current-state) for the latest checkpoint, [milestone history](PROGRESS.md#milestone-history) for earlier evidence, [helper-script index](scripts/guide-02/README.md) for runnable tooling, and [roadmap](ROADMAP.md) for future topics.

## Guides and agendas

<a id="agenda-from-physical-gpu-to-application"></a>
<a id="guides"></a>

### Guide 01 — GPU Node Validation

[Open Guide 01](docs/guides/01-brev-gpu-node-validation.md)

**Goal:** Understand the path from the provisioned GPU to a working application on an NVIDIA L4 Brev node. Follow these eight layers as the reading agenda; use the minimum relevant check during customer troubleshooting.

1. [Physical GPU](docs/guides/01-brev-gpu-node-validation.md#1-physical-gpu) — identify the provisioned GPU.
2. [PCIe enumeration](docs/guides/01-brev-gpu-node-validation.md#2-pcie-enumeration) — check device visibility with `lspci`.
3. [Linux kernel](docs/guides/01-brev-gpu-node-validation.md#3-linux-kernel) — identify kernel, OS, and architecture.
4. [NVIDIA driver](docs/guides/01-brev-gpu-node-validation.md#4-nvidia-driver) — use `nvidia-smi`; its CUDA field reports driver compatibility, not the installed Toolkit/runtime version.
5. [CUDA runtime / Toolkit](docs/guides/01-brev-gpu-node-validation.md#5-cuda-runtime--toolkit) — inspect `nvcc` in a devel container; a host Toolkit is not required for this setup.
6. [Container runtime](docs/guides/01-brev-gpu-node-validation.md#6-container-runtime) — validate `docker info` and GPU-enabled `docker run`.
7. [PyTorch / NeMo](docs/guides/01-brev-gpu-node-validation.md#7-pytorch--nemo) — check framework CUDA access and model placement.
8. [Application](docs/guides/01-brev-gpu-node-validation.md#8-application) — process real inputs and verify the output; continue into Guide 02.

**Status:** Host/driver visibility, container GPU access, compiler inspection, and framework access are validated for the reported checks. Guide 02 now verifies one pretrained NSC GPU transcription. Standalone CUDA kernel/tensor checks, training, and benchmarks remain pending; Guide 01 is in progress.

### Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Open Guide 02](docs/guides/02-nemotron-streaming-singapore-english.md)

**Goal:** Fine-tune `nvidia/nemotron-3.5-asr-streaming-0.6b` on NSC Part 6 using the NVIDIA L4 Brev instance. This small POC follows **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**.

**Start here:** Read the [data strategy](docs/guides/02-nemotron-streaming-singapore-english.md#data-strategy-and-experiment-overview), then follow [node preflight](docs/guides/02-nemotron-streaming-singapore-english.md#node-preflight--complete), [NeMo/model setup](docs/guides/02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete), and [NSC download](docs/guides/02-nemotron-streaming-singapore-english.md#nsc-trainingquery-and-development-data--downloaded-and-extracted). The [guide section map](docs/guides/02-nemotron-streaming-singapore-english.md#guide-sections) links the detailed reading order; the hands-on agenda continues below.

1. [Verify source manifests on the node](docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) — counts checked; full-source audio integrity not audited.
2. [Verify speaker and utterance-ID separation](docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) — complete for train/validation; NSC test checks pending.
3. [Inspect and count transcript annotation tokens](docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-audit--complete) and [measure POC eligibility](docs/guides/02-nemotron-streaming-singapore-english.md#poc-eligibility-impact--complete) — complete; annotation policy decided.
4. [Create deterministic speaker-aware POC subsets](docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) — 300/50 generated; ownership, counts, tags, speakers, and separation independently checked.
5. [Normalize derived transcripts](docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-normalization--complete) — complete: noise-tag cleanup and whitespace normalization preserve spoken words and metadata; final scoring policy pending.
6. [Convert to Nemotron manifests](docs/guides/02-nemotron-streaming-singapore-english.md#nemotron-compatible-manifest-conversion--complete) — complete: five fields, 300/50 records, selected audio paths found, and one record per split inspected.
7. [Run one validation utterance on GPU](docs/guides/02-nemotron-streaming-singapore-english.md#single-utterance-pretrained-gpu-inference--complete) — complete on October 9, 2026; the fixed 50-utterance baseline/WER is next under recorded scoring/decoding settings.
8. Fine-tune on train only, starting with a training smoke test.
9. Use validation for development/model selection.
10. Freeze the model/configuration, normalization, and decoding settings.
11. Evaluate `nsc_test` for final held-out Singapore-English results.
12. Evaluate `gigaspeech_test` for external/OOD generalization and regressions after the NSC test.

**Status:** Setup, source checks, POC subset preparation/validation, transcript normalization, five-field Nemotron manifest conversion, and one pretrained GPU transcription are complete for the reported checks. Selected subsets have **300 train utterances / 111 speakers / ~0.66 h** and **50 validation utterances / 50 speakers / ~0.11 h**, using seed 42, no `<unk>`, and no shared speakers/IDs. Conversion reported 300/50 records and found selected audio files; one final record per split was inspected. These preparation checks establish records and paths. One pretrained validation recording was subsequently transcribed on GPU; all-50-record decoding and training-loader compatibility remain unverified. Ownership correction covered only the original subset files; new-output ownership was not reported.

**Current checkpoint (October 9, 2026):** Pretrained Nemotron inference on one NSC validation recording is verified on the L4, with model device `cuda:0`: reference `uh correct` → raw prediction `Uh correct. <en-US>`. The [guide](docs/guides/02-nemotron-streaming-singapore-english.md#single-utterance-pretrained-gpu-inference--complete) records the initial prompt error and successful explicit non-Lhotse configuration. Full validation baseline/WER, training, NSC test, and GigaSpeech remain unrun; no test/benchmark downloads are documented.

**Next session:** Finalize scoring/decoding settings and run the [fixed 50-utterance pretrained validation baseline/WER](docs/guides/02-nemotron-streaming-singapore-english.md#next-50-utterance-validation-baseline-and-wer--planned). Exclude language markers from scoring while retaining raw predictions; use identical policies for later fine-tuned comparisons. Training smoke test, fine-tuning, post-training evaluation, true streaming, and latency benchmarks remain pending. Curator/manual annotation review remain future work; the generic WAV appendix remains unexecuted.

## Future guides

Storage/data access → Slurm → distributed training/NCCL → Kubernetes/GPU Operator → observability → architecture synthesis. These topics are planned in the [roadmap](ROADMAP.md); guide documents are added when work begins.

## Repository layout

```text
ai-infrastructure-guide/
├── README.md                 # Guide overview and agendas
├── ROADMAP.md                # Future topics
├── PROGRESS.md               # Current status and dated milestone history
├── .gitignore
├── scripts/
│   └── guide-02/
│       ├── README.md         # Helper purposes, inputs, and node setup
│       ├── check_split_overlap.py
│       ├── inspect_transcript_tags.py
│       ├── check_poc_eligibility.py
│       ├── create_poc_subsets.py
│       ├── normalize_transcripts.py
│       ├── convert_to_nemo_manifest.py
│       └── baseline_smoke.py
└── docs/
    └── guides/
        ├── README.md         # Short directory index
        ├── 01-brev-gpu-node-validation.md
        └── 02-nemotron-streaming-singapore-english.md
```

The [written guides](docs/guides/README.md) contain commands, architecture explanations, and recorded evidence. Runnable assets live under `scripts/guide-NN/`; the [Guide 02 tooling index](scripts/guide-02/README.md) explains six preparation helpers and the one-record GPU inference script. The inference script implements the reported successful workflow; it has not been compared byte-for-byte with the Brev copy. Repository scripts and the Brev workspace are separate locations; a commit does not deploy a helper to the node. The [progress log](PROGRESS.md) separates current status from historical milestones; the [roadmap](ROADMAP.md) holds future topics.

## Documentation standard

Distinguish observed results from prepared instructions and planned work. Record the environment, reproduction steps, troubleshooting lessons, and costs/cleanup where known. Publish reviewed, sanitized evidence; keep credentials, datasets, model caches/weights, raw recordings, and raw workload outputs outside Git. A local laptop path is not automatically available on an SSH-connected GPU host.

This is an independent learning guide and portfolio.
