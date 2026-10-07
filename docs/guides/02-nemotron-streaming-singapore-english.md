# Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md)

## Goal and scope

Build a small proof of concept (POC) and tutorial for fine-tuning `nvidia/nemotron-3.5-asr-streaming-0.6b` on Singapore National Speech Corpus (NSC) Part 6, using an NVIDIA L4 24 GB class Brev instance. The goal is a manageable learning exercise, not production-quality tuning. Model loading, dataset preparation, training, and evaluation results are documented only after they are performed.

## Where this guide fits in the agenda

[Guide 01](01-brev-gpu-node-validation.md#agenda-validate-the-stack-from-gpu-to-application) covers host, container, and framework access. This guide continues at **layer 8: application** with the Nemotron ASR POC. NeMo import and model placement extend the reported layer 7 checks; transcription, training, and evaluation remain pending.

## Guide 02 workflow agenda

[Node preflight](#node-preflight--complete) and [NeMo/model loading](#nemo-container-and-model-on-gpu--complete) are recorded setup checks. Source inspection, train/validation separation, annotation enumeration, eligibility analysis, the POC annotation policy, subset generation, and [independent subset checks](#derived-poc-subset-validation--complete) are complete. The 300 train / 50 validation records have no `<unk>` tags or shared speakers/IDs. Referenced-audio integrity, NSC test checks, and steps 5–12 remain pending.

1. [Verify source manifests on the node](#split-construction-and-hands-on-verification) — counts checked; referenced-audio integrity not yet reported.
2. [Verify speaker and utterance-ID separation](#split-construction-and-hands-on-verification) — complete for train/validation; NSC test checks pending.
3. [Inspect and count transcript annotation tokens](#transcript-annotation-audit--complete) and [measure POC eligibility](#poc-eligibility-impact--complete) — complete; annotation policy decided.
4. [Create deterministic speaker-aware POC subsets](#deterministic-poc-subset-generation--complete) — 300/50 generated; ownership, counts, tags, speakers, and separation independently checked.
5. [Normalize derived transcripts](#next-transcript-normalization--planned): remove `<v-noise>`/`<noise>` tokens while preserving actual words, local speech, and fillers; record any additional scoring rules.
6. Convert derived data to NeMo manifests with audio paths valid inside the container.
7. Run baseline inference on validation and record WER.
8. Fine-tune on train only, starting with a training smoke test on the L4.
9. Use validation for development/model selection; record changes and compare checkpoints under the same scoring rules.
10. Freeze the model/configuration: checkpoint, normalization, and decoding settings.
11. Evaluate `nsc_test` for final held-out Singapore-English results — planned.
12. Evaluate `gigaspeech_test` for external/OOD generalization and regressions after the NSC test — planned.

Perform data preparation in the SSH-connected GPU host's working directory, outside Git. Download there and bind-mount host data into the container; a laptop path is not automatically available on the remote node. Record source locations/versions, the script and selection seed, selected IDs, overlap results, normalization rules, container/model versions, and run settings. Keep validation IDs fixed; when scoring rules change during development, rescore both models consistently. Seed 42, generation output, and independent subset checks are recorded below; normalization and training settings remain pending.

This POC keeps the experiment small for learning. A customer deployment would choose data coverage and evaluation sizes around actual users, audio conditions, and acceptance criteria; ~300/50 utterances are not a production recommendation. WER checks recognition quality, while streaming behavior, latency, throughput, and scheduler integration remain separate future work.

## Data strategy and experiment overview

The intended progression is **train → validation/development loop → freeze model/configuration → NSC held-out test → GigaSpeech external/OOD benchmark**. NSC query/dev sources are downloaded and audited; **300/50 POC subsets have been generated and independently checked** for ownership, record counts, tags, speakers, and separation. Test and benchmark sources are identified in the plan, with local preparation and evaluation still pending. **Validation** and **dev** mean the same role; the source directory remains `nsc_dev_3h`.

| Stage | Planned source | POC / evaluation size | Purpose | Model learns from it? | When used | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| Train | `nsc_query_5h` | 300 utterances, independently checked | Fine-tuning | Yes, directly through gradient updates | First learning stage | Subset checks complete; normalization pending |
| Validation | `nsc_dev_3h` | 50 utterances, independently checked | Baseline comparison, tuning, checkpoint/model decisions | No gradients; influences development indirectly | Before and during fine-tuning development | Subset checks complete; normalization pending |
| Test | `nsc_test` | 3,684 utterances / ~7 h, upstream | Final held-out Singapore-English evaluation | No gradients or development tuning | After model/configuration is frozen | Planned; local preparation/evaluation pending |
| External benchmark | `gigaspeech_test` | 19,930 utterances / 35.4 h, upstream | Out-of-domain (OOD) generalization/regression check | No gradients or routine tuning | After NSC test evaluation | Planned; local preparation/evaluation pending |

```text
Train: nsc_query_5h → 300-utterance POC subset
     ↓
Gradient updates
     ↓
Validation: nsc_dev_3h → 50-utterance POC subset
     ↓
Development loop: compare models, tune choices, select checkpoint
     ↓
Freeze model and configuration
     ↓
NSC held-out test: nsc_test
     ↓
Final held-out Singapore-English evaluation
     ↓
GigaSpeech external benchmark: gigaspeech_test
     ↓
OOD generalization/regression check
```

Capture baseline **word error rate (WER)** on validation before the first fine-tuning run. Return to the same validation subset to compare checkpoints and development choices. Training and validation can repeat during development; final test evaluation comes after those decisions are settled.

### Available sources: origin and contents

The [upstream dataset documentation](https://huggingface.co/datasets/pengyizhou/IALP-2026-data) identifies the four sources below. Query/dev counts were recorded locally; test/benchmark sizes are upstream descriptions, not locally verified counts or completed evaluations.

| Source | Origin | Full source size | Contents and intended use |
| --- | --- | --- | --- |
| `nsc_query_5h` | NSC IMDA Part 6 train partition | 2,289 utterances locally; 5.01 h from manifest durations; ~214 MB, recorded as `214M` | FLAC audio and JSONL manifest; 300 training utterances generated for future fine-tuning |
| `nsc_dev_3h` | NSC IMDA Part 6 train partition | 1,316 utterances locally; 3.02 h from manifest durations | FLAC audio and JSONL manifest; 50 validation utterances generated for future development/checkpoint decisions |
| `nsc_test` | Official NSC IMDA Part 6 test partition | 3,684 utterances / ~7 h, upstream | FLAC audio and JSONL manifest, per upstream; planned held-out in-domain evaluation |
| `gigaspeech_test` | Separate GigaSpeech test corpus | 19,930 utterances / 35.4 h, upstream | WAV PCM_16 audio and JSONL manifest with normalized references, per upstream; planned external/OOD evaluation |

The downloaded query/dev JSONL records use `id`, `speaker`, `duration` (seconds), `text` (reference transcript), and `audio` (relative path into `audio/`). Their download commands and examples are below. Local train/dev hour totals sum manifest duration fields; test/benchmark hours remain upstream descriptions. Independent checks confirmed 300/50 POC records.

Current original manifests on the Brev host, unchanged:

```text
/home/ubuntu/data/nsc/nsc_query_5h/manifest.jsonl
/home/ubuntu/data/nsc/nsc_dev_3h/manifest.jsonl
```

### Split construction and hands-on verification

`nsc_query_5h` and `nsc_dev_3h` are separate selections from the NSC Part 6 train partition, constructed using different speaker sets. **`nsc_dev_3h` is not a subset of `nsc_query_5h`**. The [publisher's split-construction notes](https://huggingface.co/datasets/pengyizhou/IALP-2026-data#split-construction-nsc) state that query, dev, and `nsc_test` are mutually speaker-disjoint, and query/dev exclude every speaker appearing in the official NSC test partition. `nsc_test` comes from that official test partition.

**Verified on the Brev node:** The operator used [`check_split_overlap.py`](../../scripts/guide-02/check_split_overlap.py) to check the complete train and validation manifests. The script accepts both manifest paths at runtime and only reads them.

Rerun from the repository root **on the GPU compute node**. The checkout and scripts must exist on that node; a laptop checkout is not automatically available over SSH. These reproduction commands map the host source directory to `/data/nsc` inside the container; change the mount and arguments if your data lives elsewhere.

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v "$PWD/scripts/guide-02:/scripts:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /scripts/check_split_overlap.py \
  /data/nsc/nsc_query_5h/manifest.jsonl \
  /data/nsc/nsc_dev_3h/manifest.jsonl
```

Verified Brev output:

```text
Train records:   2289
Dev records:     1316
Train speakers:  111
Dev speakers:    64
Speaker overlap: 0
ID overlap:      0
```

The complete train and validation manifests share no speakers or utterance IDs; dev is not simply a subset of query. This independently confirms the upstream train/dev speaker-disjoint design. It does **not** verify separation from `nsc_test`: compare query/dev/test pairwise when preparing test. The derived POC subsets were also [checked independently](#derived-poc-subset-validation--complete).

**SA/reproducibility:** Both checks ran inside the version-pinned NeMo Speech container with the source dataset mounted read-only. In these commands, `:ro` makes the data and script mounts read-only; `/scripts` is the container's script directory, and manifest paths are command-line arguments. Python comes from the container, without installing it on the host or modifying the dataset. Neither helper contains Brev-specific hardcoded paths. Keep the source version/location and script version with the run notes; the download directory below remains a reproduction example.

### Direct learning versus development decisions

Training data changes model weights through backpropagation. Validation data must never enter gradient updates, but its results still influence the final system indirectly: an engineer may choose a learning rate, checkpoint, normalization rule, or decoding setting because it improves validation WER. Finalize those choices using validation, and record them before testing.

Test results are for evaluating the finalized internal result. If we repeatedly tune choices based on test results, that set effectively becomes another validation set and loses its role as an independent final check. A new untouched holdout would be needed for a fresh final evaluation. See the [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) for this distinction.

### What results can tell us

- **Train:** Lower training loss shows better fit to training examples; it does not establish performance on unseen speech.
- **Validation:** WER on the fixed ~50 utterances supports baseline comparisons and development decisions under recorded scoring rules. It is a small, development-influenced result, not an unbiased final estimate for all Singapore English.
- **Test:** `nsc_test` evaluates the frozen system on held-out NSC in-domain speech. Conclusions apply to this test distribution; they do not establish production performance or generalization to every customer.
- **External benchmark:** `gigaspeech_test` checks generalization and regressions on a separate corpus. Compare baseline and final-model results under fixed scoring rules, without routine tuning on this benchmark. Its results do not establish universal generalization.

### Evaluation preparation — TODO

- **NSC test:** Prepare `nsc_test` on the GPU host, verify local counts and overlap, and keep it out of the development loop. Evaluate only after the model, normalization, and decoding configuration are frozen.
- **GigaSpeech OOD benchmark:** Prepare `gigaspeech_test` on the host and document local counts and scoring/reference normalization. Run the planned generalization/regression comparison after NSC test evaluation. Preparation and both evaluations remain unvalidated.

## Node preflight — complete

**Milestone recorded:** 2026-10-05, the documentation sync date. The following checks were performed on the GPU node before pulling the NeMo container or downloading the model/data. Evidence is the operator's sanitized summary.

```bash
df -h
free -h
nvidia-smi
```

| Resource or check | Observed value |
| --- | --- |
| Root filesystem total | 267 GB, as reported in the milestone summary |
| Root filesystem used | 53 GB |
| Root filesystem available | 214 GB |
| Host RAM total | 15 GiB |
| Host RAM available | Approximately 14 GiB |
| Swap | None configured |
| GPU | NVIDIA L4 |
| GPU VRAM total / used | 23,034 MiB / 0 MiB |
| GPU utilization | 0% |
| NVIDIA driver | `595.91.07` |
| Driver-reported CUDA compatibility level | `13.2` |
| Running GPU processes | None reported |

Disk figures preserve the operator's reported units and rounded values; GNU `df -h` uses powers of 1024 for human-readable sizes. This is one preflight snapshot, not a measurement under model or training load.

### What the checks establish

- `df -h` checks filesystem capacity and usage; `-h` formats sizes in human-readable units. Check the filesystem where downloads and container storage will live. See the [GNU df manual](https://www.gnu.org/software/coreutils/manual/html_node/df-invocation.html).
- `free -h` checks CPU/system RAM and swap, separately from GPU VRAM. Linux uses otherwise unused RAM for filesystem cache, so low `free` memory can coexist with high `available` memory. `available` estimates memory usable by new applications without swapping, accounting for reclaimable cache. See the [free manual](https://www.man7.org/linux/man-pages/man1/free.1.html).
- `nvidia-smi` checks GPU/driver communication and reports GPU memory, utilization, thermals, power, and active processes. No temperature or power reading was supplied for this milestone. The reported CUDA compatibility level does not establish an installed host Toolkit version. See [NVIDIA's nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Preflight checks help rule out simple causes such as disk exhaustion, current host-memory pressure, or an already-occupied GPU before investigating model/container issues. This snapshot showed available disk and RAM and an idle GPU; it does not yet establish resource sufficiency for the selected model, data, or training configuration.

## NeMo container and model on GPU — complete

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator validated the official NeMo Speech container on the Linux GPU host:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo; print(nemo.__version__)"
```

The reported NeMo version was `3.0.0`. The operator then successfully restored the model with `ASRModel.from_pretrained(...)` and configured the persistent cache:

```bash
mkdir -p ~/hf-cache

docker run --rm --gpus all \
  -v ~/hf-cache:/root/.cache/huggingface \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo.collections.asr as n; m=n.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b'); m.cuda(); print(next(m.parameters()).device)"
```

The model loaded as `EncDecRNNTBPEModelWithPrompt`. The final output was:

```text
cuda:0
```

This validates Docker GPU passthrough, the NeMo Speech stack for importing ASR and loading this model, and PyTorch/CUDA placement of model parameters on GPU 0. Earlier CUDA-container checks established the separation between the host NVIDIA driver and container CUDA userspace; this step extends that path to Nemotron model loading.

The bind mount stores Hugging Face cache files in `~/hf-cache` on the host, outside the disposable container. Removing the container with `--rm` leaves those files in place for later runs. A second-run cache hit was not separately reported. The model itself fits on the L4 for this loading check. Model placement alone does not establish successful audio inference or sufficient memory for training.

## NSC training/query and development data — downloaded and extracted

**Milestone recorded:** 2026-10-05, the documentation sync date. The operator downloaded the small NSC Part 6 training/query dataset and the separate development split from [IALP-2026-data on Hugging Face](https://huggingface.co/datasets/pengyizhou/IALP-2026-data). The counts and records below are operator-reported observations. Keep downloaded archives and audio on the GPU host, outside the repository.

For reproduction, first choose a host working directory outside the Git checkout. Both archives extract into the current directory. For example:

```bash
mkdir -p ~/asr-data
cd ~/asr-data
```

### Training/query split

```bash
wget https://huggingface.co/datasets/pengyizhou/IALP-2026-data/resolve/main/nsc-query.tar.gz
tar xzf nsc-query.tar.gz
wc -l nsc_query_5h/manifest.jsonl
du -sh nsc_query_5h
```

Reported results: **2,289 manifest records** and **214M** extracted directory size (`du -sh`). The source contains approximately **5 hours** of speech; its duration was not independently summed in this milestone.

```text
nsc_query_5h/
├── audio/
├── manifest.jsonl
├── text
├── utt2spk
└── wav.scp
```

Example training record (ID and audio filename shortened):

```json
{"id":"...","speaker":"00038","duration":2.58,"text":"call one telco","audio":"audio/...flac"}
```

### Separate development split

```bash
wget https://huggingface.co/datasets/pengyizhou/IALP-2026-data/resolve/main/nsc-dev.tar.gz
tar xzf nsc-dev.tar.gz
wc -l nsc_dev_3h/manifest.jsonl
```

Reported result: **1,316 manifest records**, with approximately **3 hours** of speech in the source. Its duration was not independently summed in this milestone. This is our validation source, not an already selected final test set.

Example development record (ID and audio filename shortened):

```json
{"id":"...","speaker":"00017","duration":6.24,"text":"okay sure <v-noise> uh good afternoon may i have your contact number in case the line like get uh disconnected","audio":"audio/...flac"}
```

The inspected train/validation records share the fields `id`, `speaker`, `duration`, `text`, and `audio`, with relative FLAC paths. These source manifests have not yet been converted to NeMo format.

<a id="transcript-annotations-and-normalization--planned"></a>

### Transcript annotation audit — complete

**Verified observation:** The operator inspected both complete source manifests for `<...>` annotation tokens using [`inspect_transcript_tags.py`](../../scripts/guide-02/inspect_transcript_tags.py). It accepts one or more manifest paths, scans transcript text, and reports tagged records separately from individual token occurrences.

From the same repository root on the compute node:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v "$PWD/scripts/guide-02:/scripts:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /scripts/inspect_transcript_tags.py \
  /data/nsc/nsc_query_5h/manifest.jsonl \
  /data/nsc/nsc_dev_3h/manifest.jsonl
```

Verified Brev results, summarized by source (the script also prints each manifest path):

```text
Train: nsc_query_5h
Total records:      2289
Records with tags:   511
<v-noise>:           598
<unk>:               145
<noise>:              49
```

```text
Validation: nsc_dev_3h
Total records:      1316
Records with tags:   286
<v-noise>:           345
<unk>:               104
<noise>:              24
```

Tags are common in both sources. Token counts are occurrences, not distinct records; a transcript can contain several tags. The reported audit found these three types in the checked manifests, rather than only `<v-noise>`.

| Token | Intended NSC annotation meaning |
| --- | --- |
| `<v-noise>` | Vocal/non-lexical noise annotation, not an ordinary spoken word |
| `<noise>` | Non-vocal/background noise annotation |
| `<unk>` | Unclear/unidentified speech under the NSC transcription convention |

**Data-quality caveat:** The intended definition does not guarantee that every individual `<unk>` label is correct. Transcript context suggests that some occurrences could represent identifiable Singapore-English/Singlish speech, such as discourse particles or locally accented speech. This is a hypothesis: **we have not listened to the source audio to verify it**. It does not establish that `<unk>` means Singlish or that the annotations are wrong.

**Current POC decision:** We will not manually listen to, relabel, or correct individual `<unk>` occurrences in this tutorial. Such corrections could introduce subjective labels and reduce reproducibility; the scope remains a simple infrastructure/fine-tuning POC. Preserve the original source manifests unchanged.

**Future data-quality work:** Sample utterances containing `<unk>` → listen to source audio → categorize genuinely unintelligible speech versus possible transcription errors → optionally create a reviewed/corrected derived dataset. This work has not been performed.

### POC eligibility impact — complete

The operator validated [`check_poc_eligibility.py`](../../scripts/guide-02/check_poc_eligibility.py) to measure what would remain if whole utterances containing `<unk>` were excluded. It only reads manifests; it does not create a filtered dataset.

Exact command used from the Brev host:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/check_poc_eligibility.py \
    /data/nsc/nsc_query_5h/manifest.jsonl \
    /data/nsc/nsc_dev_3h/manifest.jsonl
```

`/home/ubuntu/data/nsc` is mounted read-only as `/data/nsc`; `/home/ubuntu/work/nemotron-poc` is mounted read-only as `/work`. The script was present in that host work directory for this run; its repository location is `scripts/guide-02/`. Manifest paths are supplied at runtime, with no hardcoded host paths in the script. No GPU access is required for this CPU/data-integrity task.

Observed output:

```text
Manifest: /data/nsc/nsc_query_5h/manifest.jsonl
Total records:       2289
Records with <unk>:  131
Records remaining:   2158
Total duration:      5.01 hours
<unk> duration:      0.42 hours
Remaining duration:  4.59 hours

Manifest: /data/nsc/nsc_dev_3h/manifest.jsonl
Total records:       1316
Records with <unk>:  91
Records remaining:   1225
Total duration:      3.02 hours
<unk> duration:      0.31 hours
Remaining duration:  2.71 hours
```

These figures describe the eligible source pools; the read-only check creates no filtered files. Subsequent subset generation is documented below. The check counts each `<unk>` utterance once, explaining why 131/91 affected records differ from the earlier 145/104 token occurrences.

#### POC annotation policy — decision complete

| Item | Current POC policy |
| --- | --- |
| `<unk>` | Exclude the whole utterance from the train/validation POC |
| `<v-noise>`, `<noise>` | Keep the utterance; remove the annotation token later during transcript normalization |
| Local speech such as `lah`, `wah`, `ya`, `mm` | Keep as normal speech |
| Original manifests | Never modify; write derived data separately |

Excluding `<unk>` is a pragmatic tutorial choice, not a claim that these utterances are bad data. The tag indicates speech was not confidently transcribed; deleting only the token could leave spoken content without corresponding text and create audio/text misalignment. We are deliberately avoiding manual relabeling. The eligible **2,158 train records / 4.59 h** and **1,225 validation records / 2.71 h** comfortably exceed the chosen 300/50 POC sizes.

The earlier manual-audio-review note remains future work for a more rigorous data-quality project. No individual annotations are corrected in this POC. The generator below excludes `<unk>` utterances; noise-token normalization remains pending.

#### Transcript normalization — planned

Annotation handling is decided above. Finalize and record the remaining transcript/scoring rules, then apply the policy consistently and reproducibly to separate derived files before NeMo conversion or WER. Normalized manifests have not been created; keep original manifests unchanged.

### Why keep development data separate?

Use query for gradient updates and dev for validation WER, development choices, and checkpoint selection. They share the NSC train partition but have locally verified, separate speaker and utterance-ID sets. Validation influences development, so it does not replace the final `nsc_test` evaluation.

<a id="next-create-tiny-deterministic-poc-subsets--planned"></a>
<a id="next-data-preparation-on-the-compute-node--planned"></a>

## Deterministic POC subset generation — complete

The operator successfully ran [`create_poc_subsets.py`](../../scripts/guide-02/create_poc_subsets.py) on the Brev node. It reads source manifests, excludes whole `<unk>` utterances, and writes selected source-format records to a separate output directory. Transcripts are not normalized or converted to NeMo format by this script.

### Design decisions

The eligible pools were **2,158 train utterances / 4.59 h** and **1,225 validation utterances / 2.71 h** after excluding `<unk>`. We chose **300 train / 50 validation utterances** for fast end-to-end learning on the NVIDIA L4. This is a POC engineering choice, not a production-scale train/validation ratio.

- **Train:** Cycle across speakers, taking at most one utterance per speaker per pass, to avoid a few prolific speakers dominating the small subset.
- **Validation:** Select one utterance from each of 50 distinct speakers to maximize speaker breadth.
- **Determinism:** `seed = 42`; train uses `random.Random(seed)` and validation uses `random.Random(seed + 1)`. Separate RNG streams keep changes to one subset size from unexpectedly changing the other. Reproduction assumes the same source manifests and ordering.

### Derived-data directory and container execution

The original manifests remain unchanged. The operator created a dedicated derived-data directory on the host:

```bash
mkdir -p ~/data/nsc/poc
```

```text
~/data/nsc/
├── nsc_query_5h/          # source
├── nsc_dev_3h/            # source
└── poc/                   # derived POC manifests
    ├── train_300.jsonl
    └── dev_50.jsonl
```

Successful command from the Brev host:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc:/data/nsc:ro \
  -v /home/ubuntu/data/nsc/poc:/output:rw \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/create_poc_subsets.py \
    /data/nsc/nsc_query_5h/manifest.jsonl \
    /data/nsc/nsc_dev_3h/manifest.jsonl \
    /output \
    --train-count 300 \
    --dev-count 50 \
    --seed 42
```

| Host path | Container path | Access |
| --- | --- | --- |
| `/home/ubuntu/data/nsc` | `/data/nsc` | Read-only source data (`:ro`) |
| `/home/ubuntu/work/nemotron-poc` | `/work` | Read-only tooling (`:ro`) |
| `/home/ubuntu/data/nsc/poc` | `/output` | Read/write derived output (`:rw`) |

The only writable host bind mount is the dedicated derived-data directory. The script was present in the host work directory for this run; its repository location is `scripts/guide-02/`. Manifest and output paths are runtime arguments. No `--gpus all` is required because generation is a CPU/data task.

### Observed successful output

The generator reported:

```text
Train output:    /output/train_300.jsonl
Train records:   300
Train speakers:  111
Train duration:  0.66 hours

Dev output:      /output/dev_50.jsonl
Dev records:     50
Dev speakers:    50
Dev duration:    0.11 hours
```

Train: **300 utterances across all 111 eligible train speakers**, approximately **0.66 h / 40 minutes**. Validation: **50 utterances from 50 distinct speakers**, approximately **0.11 h / 6.6 minutes**. Durations sum manifest fields and are rounded. Record and speaker counts were independently checked below.

<a id="next-validate-derived-subsets--pending"></a>

## Derived POC subset validation — complete

The operator independently checked the generated manifests on the Brev node, rather than relying on the generator's report. These checks cover file ownership, record counts, annotation tags, speakers, and utterance-ID separation; they do not establish audio integrity or model performance.

### Host-side file ownership and line counts

```bash
ls -lh ~/data/nsc/poc
```

Initial ownership in this run:

```text
-rw-r--r-- 1 root root 13K ... dev_50.jsonl
-rw-r--r-- 1 root root 82K ... train_300.jsonl
```

The container's default execution context produced root-owned output. Ownership was corrected **only for the two derived manifests**, then checked again with `ls -lh`; no original NSC files were changed:

```bash
sudo chown ubuntu:ubuntu \
  ~/data/nsc/poc/train_300.jsonl \
  ~/data/nsc/poc/dev_50.jsonl
```

Verified ownership:

```text
-rw-r--r-- 1 ubuntu ubuntu 13K ... dev_50.jsonl
-rw-r--r-- 1 ubuntu ubuntu 82K ... train_300.jsonl
```

Independent line-count check:

```bash
wc -l \
  ~/data/nsc/poc/train_300.jsonl \
  ~/data/nsc/poc/dev_50.jsonl
```

Observed:

```text
300 /home/ubuntu/data/nsc/poc/train_300.jsonl
 50 /home/ubuntu/data/nsc/poc/dev_50.jsonl
350 total
```

The tag and overlap helpers below also parsed all 300/50 records as JSONL.

### Derived annotation tags

The existing [`inspect_transcript_tags.py`](../../scripts/guide-02/inspect_transcript_tags.py) checked both derived files. To reproduce on the compute node with the helper present in the host work directory:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/inspect_transcript_tags.py \
    /data/poc/train_300.jsonl \
    /data/poc/dev_50.jsonl
```

Observed results:

```text
Manifest: /data/poc/train_300.jsonl
Total records:       300
Records with tags:    61
Tag counts:
  <v-noise>: 87
  <noise>: 5

Manifest: /data/poc/dev_50.jsonl
Total records:       50
Records with tags:    4
Tag counts:
  <v-noise>: 7
```

**`<unk>`: 0 in both subsets.** The whole-utterance exclusion worked. `<v-noise>` and sampled `<noise>` remain for later token removal; this is not normalized output.

### Derived speaker and utterance-ID separation

The existing [`check_split_overlap.py`](../../scripts/guide-02/check_split_overlap.py) independently checked the POC pair. Reproduction command:

```bash
docker run --rm \
  -v /home/ubuntu/data/nsc/poc:/data/poc:ro \
  -v /home/ubuntu/work/nemotron-poc:/work:ro \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python3 /work/check_split_overlap.py \
    /data/poc/train_300.jsonl \
    /data/poc/dev_50.jsonl
```

Observed:

```text
Train records:   300
Dev records:     50
Train speakers:  111
Dev speakers:    50
Speaker overlap: 0
ID overlap:      0
```

All 111 eligible train speakers are represented; validation contains 50 distinct speakers. The POC manifests share no speakers or utterance IDs. Both helpers use the pinned NeMo image with data/tooling mounted read-only and no GPU request; the commands above are reproduction examples. Source and derived checks remain separate evidence.

## Troubleshooting notes

### Troubleshooting: arbitrary container UID

An initial attempt added the host UID/GID option to make generated files inherit the `ubuntu` account's ownership:

```bash
--user "$(id -u):$(id -g)"
```

It first failed with:

```text
exec: python3: not found
```

The interpreter was located at `/opt/venv/bin/python3`, but invoking it under the arbitrary host UID then failed with:

```text
/opt/venv/bin/python3: Permission denied
```

Do not assume a vendor container supports arbitrary host UID/GID execution. This POC uses the image's default execution context and constrains host writes with read-only source/tooling mounts and a writable derived-output mount. The UID option is omitted from the successful command; generated-file ownership was explicitly corrected afterward as recorded above.

### Unexpected output: inspect the actual script

During node validation, `check_split_overlap.py` was accidentally overwritten with transcript-tag inspection code. Unexpected output prompted inspection of the script, preservation of the incorrect copy as `check_split_overlap.py.bad` locally, restoration of the known-good version, and rerunning validation. The incorrect artifact is not committed. Inspect the program producing unexpected output before drawing conclusions about the data.

## Next: transcript normalization — planned

Remove only `<v-noise>` and `<noise>` annotation tokens from the derived transcripts while preserving actual words, Singlish/local speech (`lah`, `wah`, `ya`, `mm`), and fillers. Write separate normalized outputs and record the policy consistently for train/validation; preserve the original source manifests and the generated subset files. Normalization has not been performed.

After normalization, convert derived data to NeMo manifests with container-valid audio paths, then run baseline validation inference. Conversion, baseline inference, fine-tuning, checkpointing, validation WER, `nsc_test`, and `gigaspeech_test` remain pending. Fixed sources, sampling settings, and seed support reproducible selection; byte-for-byte artifact testing is not a required POC step.

## Completion boundaries and next step

| Guide 02 stage | Status |
| --- | --- |
| Node resource preflight | Complete |
| NeMo container validation | Complete: NeMo 3.0.0, ASR import and model loading |
| Model download and loading | Complete: `EncDecRNNTBPEModelWithPrompt` |
| Persistent Hugging Face cache | Host directory and bind mount configured; files survive container removal |
| Model placement on GPU | Complete: `cuda:0` |
| Single-file WAV inference | Instructions prepared; GPU execution and transcript pending |
| NSC training/query download and extraction | Complete: 2,289 records, 214M |
| NSC dev download and extraction | Complete: 1,316 records |
| Four-stage data strategy and workflow | Documented; execution remains pending |
| Source train/dev manifest inspection | Complete for reported source checks; referenced-audio integrity not yet reported |
| Speaker-overlap verification | Complete for train/validation: 111 / 64 speakers, 0 overlap; NSC test checks pending |
| Utterance-ID overlap verification | Complete for train/validation: 0 overlap; NSC test checks pending |
| Annotation-token inventory and counts | Complete for both full source manifests: `<v-noise>`, `<unk>`, `<noise>` |
| `<unk>` eligibility impact analysis | Complete: 2,158 / 1,225 records would remain, 4.59 / 2.71 h |
| POC `<unk>`/annotation-handling decision | Complete: exclude whole `<unk>` utterances; keep noise-tagged utterances for later token removal |
| Deterministic speaker-aware subset generation | Complete: seed 42; generator reported 300 train / 50 validation |
| Derived POC directory creation | Complete: `~/data/nsc/poc` |
| Derived POC manifest creation | Complete: `train_300.jsonl` / `dev_50.jsonl` |
| Generated-file ownership correction | Complete: only the two derived files changed from `root:root` to `ubuntu:ubuntu` |
| Independent host-side line counts | Complete: 300 train / 50 validation |
| Derived annotation-tag validation | Complete: `<unk>` absent; train `<v-noise>` 87 / `<noise>` 5, validation `<v-noise>` 7 |
| Derived speaker/utterance-ID validation | Complete: 111 / 50 speakers; 0 shared speakers and 0 shared IDs |
| Independent derived-subset validation | Complete for the ownership, count, tag, speaker, and separation checks above |
| Transcript normalization output | Not yet created; remaining transcript/scoring rules pending |
| Annotation-policy application | Partial: generator excludes `<unk>`; noise-token normalization pending |
| NeMo manifest conversion | Not yet complete |
| Baseline inference / validation WER | Not yet complete |
| Training smoke test | Not yet complete |
| Fine-tuning | Not yet complete |
| Checkpointing | Not yet complete |
| Checkpoint/model comparison on validation | Not yet complete |
| Final model/configuration | Not yet finalized |
| NSC held-out test (`nsc_test`) | Planned; local preparation/verification and evaluation pending after configuration freeze |
| GigaSpeech external/OOD benchmark (`gigaspeech_test`) | Planned; local preparation/verification and evaluation pending after NSC test |
| Performance benchmarking | Not yet complete |
| True streaming inference | Not yet complete |
| Manual annotation/audio review | Future work; excluded from this POC |

Next, normalize derived transcripts, then convert NeMo manifests. Later: baseline validation → train → validation/development loop → freeze model/configuration → `nsc_test` → `gigaspeech_test`. Both evaluations remain planned. The infrastructure progression remains Docker validation → actual Nemotron inference → streaming inference → package the workload → SLURM; streaming and scheduler work remain future milestones.

<a id="single-file-inference-reference--not-yet-executed"></a>

## Appendix: optional single-file inference smoke test — not yet executed

These previously prepared commands remain available for a one-file check; they are not evidence of completed baseline inference. The current next milestone is transcript normalization above. Run these commands in the GPU host's shell, including when connected over SSH. They require no notebook, GUI, or microphone. This is whole-file inference with a streaming-capable model; true streaming is a later milestone.

### 1. Create an audio directory and download a small sample

```bash
mkdir -p ~/asr-audio

curl -fL \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav \
  -o ~/asr-audio/sample.wav
```

If `curl` is unavailable, use:

```bash
wget -O ~/asr-audio/sample.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

This file is linked in an [NVIDIA NeMo ASR example](https://github.com/NVIDIA-NeMo/Speech/discussions/3553). The guide maintainer verified its download and format: mono, 16 kHz, approximately 7.4 seconds, 238 kB. It is a general English smoke-test sample, not evidence of Singapore English adaptation. `curl -fL` fails on HTTP errors and follows redirects; `-o` chooses the local filename. Audio stays outside the repository.

### 2. Load the model on GPU and print the transcript

```bash
docker run --rm --gpus all -i \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/asr-audio:/audio:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python - <<'PYTHON'
import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b"
)
model.cuda()
model.eval()
print("Model device:", next(model.parameters()).device)

with torch.inference_mode():
    results = model.transcribe(
        audio=["/audio/sample.wav"],
        batch_size=1,
        return_hypotheses=True,
        target_lang="en-US",
    )

print("Transcript:", results[0].text)
PYTHON
```

The [NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) documents NeMo loading and transcription. The [prompt-conditioned NeMo model source](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/rnnt_bpe_models_prompt.py) supports the `audio`, `return_hypotheses`, and language-prompt arguments used here.

| Command part | Purpose |
| --- | --- |
| `docker run` | Starts a container on the GPU host |
| `--rm` | Removes the stopped container; host-mounted cache and audio remain |
| `--gpus all` | Makes the host GPUs available to the container |
| `-i` | Keeps standard input open so the Python code can enter the container |
| Cache `-v` mount | Reuses the persistent host Hugging Face cache |
| Audio `-v` mount | Maps `~/asr-audio` on the host to `/audio` in the container; `:ro` makes it read-only |
| `nvcr.io/nvidia/nemo-speech:26.07.00` | Reuses the NeMo Speech image already validated for model loading |
| `python -` | Runs Python code read from standard input |
| `<<'PYTHON' ... PYTHON` | Sends the multiline code without host-shell variable expansion; no interactive terminal (`-t`) is needed |

`model.cuda()` moves the model to GPU 0; `model.eval()` selects evaluation behavior, and `torch.inference_mode()` disables gradient tracking. `batch_size=1` keeps this to one file. `target_lang="en-US"` supplies the English prompt for this sample, without claiming a Singapore-specific adaptation. `return_hypotheses=True` returns a hypothesis whose `.text` is printed.

Success means the process exits normally, reports `Model device: cuda:0`, and prints a nonempty transcript after `Transcript:`. Record the actual text and any errors after running it. No transcript or inference success is claimed yet. If the file is missing, check the host download and `/audio` mount first; if model loading succeeds but transcription fails, retain the error for targeted diagnosis.
