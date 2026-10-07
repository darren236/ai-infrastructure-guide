import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_eligible_records(path):
    """
    Load records from a JSONL manifest that are eligible for this POC.

    Current POC policy:
    - Exclude utterances containing <unk>.
    - Keep utterances containing <noise> or <v-noise>.
      Those annotation tokens will be normalized in a later step.

    The source manifest is only read; it is never modified.
    """
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "<unk>" in row["text"]:
                continue

            records.append(row)

    return records


def group_by_speaker(records):
    """
    Group manifest records by speaker ID.

    Returns a dictionary like:

        {
            "speaker_1": [record, record, ...],
            "speaker_2": [record, record, ...],
        }
    """
    by_speaker = defaultdict(list)

    for row in records:
        by_speaker[row["speaker"]].append(row)

    return by_speaker


def sample_train(records, count, rng):
    """
    Select a deterministic, speaker-aware training subset.

    Instead of randomly drawing all utterances from one large pool,
    we cycle across speakers. This reduces the chance that prolific
    speakers dominate a small POC subset.
    """
    by_speaker = group_by_speaker(records)

    # Shuffle each speaker's utterances deterministically.
    for speaker_records in by_speaker.values():
        rng.shuffle(speaker_records)

    # Shuffle speaker order deterministically as well.
    speakers = list(by_speaker.keys())
    rng.shuffle(speakers)

    selected = []
    index_by_speaker = {speaker: 0 for speaker in speakers}

    # Take at most one utterance per speaker in each pass.
    # Continue making passes until we have the requested number.
    while len(selected) < count:
        added_this_round = 0

        for speaker in speakers:
            index = index_by_speaker[speaker]
            speaker_records = by_speaker[speaker]

            # Skip speakers that have no unused utterances remaining.
            if index >= len(speaker_records):
                continue

            selected.append(speaker_records[index])
            index_by_speaker[speaker] += 1
            added_this_round += 1

            if len(selected) == count:
                break

        # Fail clearly rather than silently returning too few records.
        if added_this_round == 0:
            raise ValueError(
                f"Requested {count} training records, "
                f"but only {len(selected)} eligible records were available."
            )

    return selected


def sample_validation(records, count, rng):
    """
    Select one validation utterance from each of `count` different speakers.

    For our 50-utterance validation POC, this means:

        50 utterances
        50 unique speakers

    This gives the small validation set broader speaker coverage.
    """
    by_speaker = group_by_speaker(records)
    speakers = list(by_speaker.keys())

    if len(speakers) < count:
        raise ValueError(
            f"Requested {count} validation speakers, "
            f"but only {len(speakers)} eligible speakers are available."
        )

    # Deterministically choose which speakers participate.
    rng.shuffle(speakers)
    selected_speakers = speakers[:count]

    selected = []

    for speaker in selected_speakers:
        speaker_records = by_speaker[speaker]

        # Pick one deterministic random utterance from this speaker.
        selected.append(rng.choice(speaker_records))

    return selected


def write_manifest(records, path):
    """
    Write selected records to a new JSONL manifest.

    The parent output directory must already exist.
    Source manifests remain untouched.
    """
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def duration_hours(records):
    """Return total audio duration in hours."""
    return sum(float(row["duration"]) for row in records) / 3600


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic speaker-aware train/dev subsets for the Nemotron POC."
    )

    parser.add_argument(
        "train_manifest",
        type=Path,
        help="Source training JSONL manifest",
    )

    parser.add_argument(
        "dev_manifest",
        type=Path,
        help="Source validation JSONL manifest",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where derived POC manifests will be written",
    )

    parser.add_argument(
        "--train-count",
        type=int,
        default=300,
        help="Number of training utterances to select (default: 300)",
    )

    parser.add_argument(
        "--dev-count",
        type=int,
        default=50,
        help="Number of validation utterances to select (default: 50)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling (default: 42)",
    )

    args = parser.parse_args()

    # Require the output directory to exist.
    # We deliberately do not create arbitrary directories from the script.
    if not args.output_dir.is_dir():
        raise ValueError(f"Output directory does not exist: {args.output_dir}")

    # Load only records that meet our current POC eligibility policy.
    train_records = load_eligible_records(args.train_manifest)
    dev_records = load_eligible_records(args.dev_manifest)

    # Use separate deterministic RNG streams so changing one subset's size
    # does not unexpectedly change the other subset.
    train_rng = random.Random(args.seed)
    dev_rng = random.Random(args.seed + 1)

    train_subset = sample_train(
        train_records,
        args.train_count,
        train_rng,
    )

    dev_subset = sample_validation(
        dev_records,
        args.dev_count,
        dev_rng,
    )

    train_output = args.output_dir / "train_300.jsonl"
    dev_output = args.output_dir / "dev_50.jsonl"

    write_manifest(train_subset, train_output)
    write_manifest(dev_subset, dev_output)

    # Report enough information to validate what was created.
    print(f"Train output:    {train_output}")
    print(f"Train records:   {len(train_subset)}")
    print(f"Train speakers:  {len({row['speaker'] for row in train_subset})}")
    print(f"Train duration:  {duration_hours(train_subset):.2f} hours")
    print()
    print(f"Dev output:      {dev_output}")
    print(f"Dev records:     {len(dev_subset)}")
    print(f"Dev speakers:    {len({row['speaker'] for row in dev_subset})}")
    print(f"Dev duration:    {duration_hours(dev_subset):.2f} hours")


if __name__ == "__main__":
    main()
