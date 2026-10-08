# Guide 02 helper scripts

[Repository overview](../../README.md#guides-and-agendas) · [Guide 02 section map](../../docs/guides/02-nemotron-streaming-singapore-english.md#guide-sections)

These six scripts support the operator-reported NSC checks, POC selection, transcript cleanup, and manifest conversion. They use Python's standard library and runtime path arguments. The two new scripts are preserved as supplied session versions. Follow the linked guide sections for the version-tagged NeMo-container commands and reported results; no image digest is recorded.

| Script | Inputs | Effect | Guide section |
| --- | --- | --- | --- |
| [check_split_overlap.py](check_split_overlap.py) | Train and validation JSONL manifests | Read-only: record/speaker counts and cross-split speaker/utterance-ID overlap | [Source checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) · [Derived checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-speaker-and-utterance-id-separation) |
| [inspect_transcript_tags.py](inspect_transcript_tags.py) | One or more JSONL manifests | Read-only: tagged-record and annotation-token counts | [Source audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-audit--complete) · [Derived audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-annotation-tags) |
| [check_poc_eligibility.py](check_poc_eligibility.py) | One or more JSONL manifests | Read-only: record/duration impact of excluding whole `<unk>` utterances | [Eligibility analysis](../../docs/guides/02-nemotron-streaming-singapore-english.md#poc-eligibility-impact--complete) |
| [create_poc_subsets.py](create_poc_subsets.py) | Source train/dev manifests, existing output directory; defaults: 300/50 records, seed 42 | Reads sources; writes `train_300.jsonl` and `dev_50.jsonl` to the output directory and reports counts/durations | [Subset generation](../../docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) |
| [normalize_transcripts.py](normalize_transcripts.py) | Source-format POC manifest and new output path | Removes approved noise tags, collapses whitespace, preserves other metadata; refuses unexpected tags, empty transcripts, and existing outputs | [Normalization](../../docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-normalization--complete) |
| [convert_to_nemo_manifest.py](convert_to_nemo_manifest.py) | Normalized manifest, original dataset root, new output path; `--language en-US` | Checks selected audio-file existence and containment; writes only `audio_filepath`, `duration`, `text`, `lang`, `target_lang`; refuses existing output | [Nemotron conversion](../../docs/guides/02-nemotron-streaming-singapore-english.md#nemotron-compatible-manifest-conversion--complete) |

## Scripts and host paths

The reported Brev runs used `/home/ubuntu/work/nemotron-poc` for executed scripts, separate from `scripts/guide-02/` in this repository and from source/derived data. A checkout on the compute node is not confirmed. A GitHub commit does not deploy files there, and a laptop checkout is not automatically available over SSH. The operator created the new scripts with `vim` in the host workspace, as recorded in the guide.

If a checkout is available **on the node**, you may stage its helpers from that checkout's repository root. This is a setup example, not a confirmed deployment or additional reported node run:

```bash
mkdir -p ~/work/nemotron-poc
cp scripts/guide-02/*.py ~/work/nemotron-poc/
```

The guide uses two script mounts: a node checkout's `scripts/guide-02` mounted as `/scripts`, or the separate work directory mounted as `/work`. Both are read-only. Manifest arguments use paths inside the container; the bind mounts connect those paths to host files.

The first three helpers only inspect manifests. The generator writes source-format subsets; normalization changes transcript text while retaining other fields. Conversion maps fields and resolves paths against the original dataset root, not the normalized-manifest directory. It omits `id`/`speaker`; use earlier POC/normalized files for overlap checks, not final five-field files. Both new scripts create outputs exclusively and must not overwrite validated artifacts.

None of these helpers decodes or modifies audio, runs ASR, or calculates WER. Conversion checks file existence, not audio properties or NeMo loader compatibility. These preparation tasks requested no GPU and used container Python without host installation. New normalized/NeMo-file ownership was not checked/corrected in the reported evidence. A later tag-inspection attempt failed because that helper was absent from the runtime workspace; repository presence is not deployment proof.

Keep datasets, generated manifests, caches, and mistaken local script copies outside Git. [The next task](../../docs/guides/02-nemotron-streaming-singapore-english.md#next-single-utterance-gpu-inference--planned) is a one-validation-utterance GPU smoke test. `baseline_smoke.py` saving/execution is unconfirmed; it is not published here as a validated helper.
