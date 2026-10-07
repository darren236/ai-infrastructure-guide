# Guide 02 helper scripts

[Repository overview](../../README.md#guides-and-agendas) · [Guide 02 section map](../../docs/guides/02-nemotron-streaming-singapore-english.md#guide-sections)

These four scripts support the recorded NSC manifest checks and POC subset preparation. They use Python's standard library and accept paths at runtime. Follow the linked guide sections for the pinned NeMo-container commands and observed results.

| Script | Inputs | Effect | Guide section |
| --- | --- | --- | --- |
| [check_split_overlap.py](check_split_overlap.py) | Train and validation JSONL manifests | Read-only: record/speaker counts and cross-split speaker/utterance-ID overlap | [Source checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#split-construction-and-hands-on-verification) · [Derived checks](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-speaker-and-utterance-id-separation) |
| [inspect_transcript_tags.py](inspect_transcript_tags.py) | One or more JSONL manifests | Read-only: tagged-record and annotation-token counts | [Source audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#transcript-annotation-audit--complete) · [Derived audit](../../docs/guides/02-nemotron-streaming-singapore-english.md#derived-annotation-tags) |
| [check_poc_eligibility.py](check_poc_eligibility.py) | One or more JSONL manifests | Read-only: record/duration impact of excluding whole `<unk>` utterances | [Eligibility analysis](../../docs/guides/02-nemotron-streaming-singapore-english.md#poc-eligibility-impact--complete) |
| [create_poc_subsets.py](create_poc_subsets.py) | Source train/dev manifests, existing output directory; defaults: 300/50 records, seed 42 | Reads sources; writes `train_300.jsonl` and `dev_50.jsonl` to the output directory and reports counts/durations | [Subset generation](../../docs/guides/02-nemotron-streaming-singapore-english.md#deterministic-poc-subset-generation--complete) |

## Scripts and host paths

Run on the SSH-connected compute node using a checkout available on that node. A laptop checkout is not automatically available remotely. The recorded Brev runs used `/home/ubuntu/work/nemotron-poc` for scripts; this directory is separate from both source data and derived output.

If you choose that work-directory layout, stage the helpers from the repository root of the checkout **on the node**. This is a setup example, not an additional reported node run:

```bash
mkdir -p ~/work/nemotron-poc
cp scripts/guide-02/*.py ~/work/nemotron-poc/
```

The guide uses two script mounts: a node checkout's `scripts/guide-02` mounted as `/scripts`, or the separate work directory mounted as `/work`. Both are read-only. Manifest arguments use paths inside the container; the bind mounts connect those paths to host files.

The first three helpers only inspect manifests. The generator writes source-format subsets to the dedicated derived directory; it does not normalize transcripts or convert to NeMo format. None of these helpers reads audio, runs ASR, or calculates WER. These data tasks need no GPU request or host Python installation when run in the pinned container.

Keep datasets, generated manifests, caches, and mistaken local script copies outside the repository. The next normalization step is planned in [Guide 02](../../docs/guides/02-nemotron-streaming-singapore-english.md#next-transcript-normalization--planned).
