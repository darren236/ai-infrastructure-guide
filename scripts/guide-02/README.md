# Guide 02 helper scripts

[Repository overview](../../README.md#guides-and-agendas) · [Guide 02 agenda](../../docs/guides/02-nemotron-streaming-singapore-english.md#hands-on-agenda) · [Current stage 8](../../docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development)

These seven Python helpers and one Bash launcher cover operator-reported NSC preparation, pretrained GPU inference, and official training. The six preparation helpers use Python's standard library; `baseline_smoke.py` uses PyTorch and NeMo inside the existing container. The Python helpers accept runtime manifest/path arguments. The Bash launcher selects smoke/full mode and uses the reported host directory layout. Normalization/conversion retain the supplied session versions; the smoke script implements the reported successful workflow, without a byte-for-byte comparison against Brev. Follow the linked guide sections for the version-tagged NeMo-container commands and reported results; no image digest is recorded. `run_nsc_train.sh` is reconstructed from supplied command history, not byte-verified against Brev or independently GPU-tested here.

The stage numbers below match the README and Guide 02's numbered sections. Run source checks in **stage 4**, derived-input preparation/checks in **stage 5**, pretrained inference in **stage 6**, and official training in **stage 7**. **Stage 8 is next:** the exact baseline evaluator source and its checkpoint-path adaptation are still pending; another training run is not the current task. Stages 1–3 use guide commands, and stages 9–10 have no executed tooling in this repository.

| Stage | Script | Inputs | Effect | Guide section |
| --- | --- | --- | --- | --- |
| 4 / 5 | [check_split_overlap.py](check_split_overlap.py) | Train and validation JSONL manifests | Read-only: record/speaker counts and cross-split speaker/utterance-ID overlap | [Source checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) · [Derived checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-speaker-and-utterance-id-separation) |
| 4 / 5 | [inspect_transcript_tags.py](inspect_transcript_tags.py) | One or more JSONL manifests | Read-only: tagged-record and annotation-token counts | [Source audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-audit--complete) · [Derived audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-annotation-tags) |
| 4 | [check_poc_eligibility.py](check_poc_eligibility.py) | One or more JSONL manifests | Read-only: record/duration impact of excluding whole `<unk>` utterances | [Eligibility analysis](../../docs/guides/02-nemotron-streaming-singapore-english.md#poc-eligibility-impact--complete) |
| 5 | [create_poc_subsets.py](create_poc_subsets.py) | Source train/dev manifests, existing output directory; defaults: 300/50 records, seed 42 | Reads sources; writes `train_300.jsonl` and `dev_50.jsonl` to the output directory and reports counts/durations | [Subset generation](../../docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) |
| 5 | [normalize_transcripts.py](normalize_transcripts.py) | Source-format POC manifest and new output path | Removes approved noise tags, collapses whitespace, preserves other metadata; refuses unexpected tags, empty transcripts, and existing outputs | [Normalization](../../docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-normalization--complete) |
| 5 | [convert_to_nemo_manifest.py](convert_to_nemo_manifest.py) | Normalized manifest, original dataset root, new output path; `--language en-US` | Checks selected audio-file existence and containment; writes only `audio_filepath`, `duration`, `text`, `lang`, `target_lang`; refuses existing output | [Nemotron conversion](../../docs/guides/02-nemotron-streaming-singapore-english.md#nemotron-compatible-manifest-conversion--complete) |
| 6 | [baseline_smoke.py](baseline_smoke.py) | Prepared validation JSONL; first recording uses `audio_filepath`, `text`, `target_lang` | Restores pretrained model on GPU, transcribes one recording with explicit non-Lhotse prompt configuration, prints raw reference/prediction; no WER | [Verified inference](../../docs/guides/02-nemotron-streaming-singapore-english.md#single-utterance-pretrained-gpu-inference--complete) |
| 7 | [run_nsc_train.sh](run_nsc_train.sh) | `smoke` or `full <steps>`; pinned NVIDIA source, 300/50 manifests/audio, cached pretrained export | Runs official NeMo training; writes checkpoints/model export and console logs to persistent host output | [Two-step training](../../docs/guides/02-nemotron-streaming-singapore-english.md#official-nemo-fine-tuning--two-step-gpu-smoke-test) |

## Official training launcher

**Stage 7 — training smoke test and artifacts.** `run_nsc_train.sh` uses NVIDIA's `speech_to_text_finetune.py` and the prompt-aware streaming YAML from **Speech v3.0.0**, supplied by the host source mount. It does not edit upstream code/configuration or implement a custom training loop. The repository version is **reconstructed from the supplied command history**, including `+trainer.limit_val_batches` and `'~model.optim.sched'`. Its command lines retain the reported settings; bytes have not been compared with the executed Brev file.

Required directories **on the GPU host**:

| Host path | Required input / use |
| --- | --- |
| `~/work/nemo-speech-src` | Official Speech checkout at `v3.0.0`; includes the training entry point and YAML |
| `~/work/nemotron-poc` | Runtime launcher workspace; writable parent `results` directory |
| `~/data/nsc` | Original audio and prepared `poc/nemo/train_300.jsonl` / `dev_50.jsonl` with container-visible audio paths |
| `~/hf-cache` | Already-downloaded pretrained `.nemo` under the launcher's observed snapshot path |

Docker and NVIDIA GPU access are prerequisites already validated in the guides. Source, data, and cache are mounted read-only. Outputs go to `~/work/nemotron-poc/results/official_finetune/<run-id>/`, with `console.log` and `nemotron_nsc/<run-id>/checkpoints/` containing training checkpoints and model export. Outputs persist after `--rm` and stay outside Git.

After staging the launcher on the node, usage is:

```bash
bash ~/work/nemotron-poc/run_nsc_train.sh smoke
```

The operator completed this mode on October 9, 2026 as `smoke-2-20261009T092152Z`: two optimizer steps, two validation batches, and saved artifacts. The repo copy has only local syntax/fidelity checks, not an independent GPU run. The snapshot path is pinned for this recorded run; verify it exists in the mounted cache before using the launcher on a new node.

Longer-run example **not executed; not the next milestone**:

```bash
bash ~/work/nemotron-poc/run_nsc_train.sh full 600
```

Both modes use the same core training route. Smoke uses `max_steps=2`, `val_check_interval=2`, `limit_val_batches=2`, and stepwise logging. Full takes a requested step count, uses fractional `1.0` for validation interval/batch limit, and logs every ten steps. A two-step run does not establish full-run capacity or consumption of all 300 recordings.

The source clone, mount diagnostic, root-owned results fix, Hydra semantics, scheduler decision, and observed artifact sizes are documented in the [training section](../../docs/guides/02-nemotron-streaming-singapore-english.md#official-nemo-fine-tuning--two-step-gpu-smoke-test). The next action is [stage 8 — checkpoint comparison and development](../../docs/guides/02-nemotron-streaming-singapore-english.md#8-checkpoint-comparison-and-development): restore/evaluate the existing export before another training run.

## Scripts and host paths

The reported Brev runs used `/home/ubuntu/work/nemotron-poc` for executed scripts, separate from `scripts/guide-02/` in this repository and from source/derived data. A checkout on the compute node is not confirmed. A GitHub commit does not deploy files there, and a laptop checkout is not automatically available over SSH. The operator created normalization/conversion scripts with `vim` in the host workspace. The October 9 inference run explicitly used `/home/ubuntu/work/nemotron-poc/baseline_smoke.py`; the repo copy implements its reported workflow/configuration, without claiming identical bytes.

If a checkout is available **on the node**, you may stage its helpers from that checkout's repository root. This is a setup example, not a confirmed deployment or additional reported node run:

```bash
mkdir -p ~/work/nemotron-poc
cp scripts/guide-02/*.py ~/work/nemotron-poc/
cp scripts/guide-02/run_nsc_train.sh ~/work/nemotron-poc/
```

The guide uses two script mounts: a node checkout's `scripts/guide-02` mounted as `/scripts`, or the separate work directory mounted as `/work`. Both are read-only. Manifest arguments use paths inside the container; the bind mounts connect those paths to host files.

The first three helpers only inspect manifests. The generator writes source-format subsets; normalization changes transcript text while retaining other fields. Conversion maps fields and resolves paths against the original dataset root, not the normalized-manifest directory. It omits `id`/`speaker`; use earlier POC/normalized files for overlap checks, not final five-field files. Normalization and conversion create outputs exclusively and must not overwrite validated artifacts.

The six preparation helpers do not decode/modify audio or run ASR. `baseline_smoke.py` performs actual one-record transcription and prints raw text, including language markers; it does not modify dataset files or calculate WER. Conversion checks file existence, not audio properties or NeMo loader compatibility. Preparation requested no GPU; the inference command uses `--gpus all`. Both use container Python without host installation. New normalized/NeMo-file ownership was not checked/corrected in the reported evidence. A later tag-inspection attempt failed because that helper was absent from the runtime workspace; repository presence is not deployment proof.

Keep datasets, generated manifests, caches, and mistaken local script copies outside Git. Both one-record inference and the [50-record offline baseline](../../docs/guides/02-nemotron-streaming-singapore-english.md#pretrained-50-utterance-validation-baseline--complete) are verified from October 9 operator reports. Baseline aggregate: 793 reference words, 10.84% WER. The executed `baseline_eval.py` lives only in the Brev workspace; add its exact source when available, not an independently reconstructed equivalent. Its raw/normalized predictions stay outside Git. [Official two-step training and saving](../../docs/guides/02-nemotron-streaming-singapore-english.md#official-nemo-fine-tuning--two-step-gpu-smoke-test) are now complete from operator evidence. Next is [restoring/evaluating that export](../../docs/guides/02-nemotron-streaming-singapore-english.md#next-restore-and-evaluate-the-exported-checkpoint--planned), preserving fixed validation and baseline policies. The optional evaluator `--model-path` change remains pending.
