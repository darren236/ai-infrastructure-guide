import argparse
import json
from pathlib import Path


def inspect_manifest(path):
    """
    Measure how much data would remain if utterances containing <unk>
    were excluded from the POC.

    This script is read-only:
    - it does not modify the manifest
    - it does not create a filtered dataset

    Current proposed POC policy:
    - <unk>     -> exclude the utterance
    - <noise>   -> keep for now
    - <v-noise> -> keep for now
    """

    # Total number of utterances in the source manifest.
    total_records = 0

    # Number of utterances containing at least one <unk>.
    unk_records = 0

    # Duration of all source audio.
    total_duration = 0.0

    # Duration of audio belonging to <unk> utterances.
    unk_duration = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # Ignore blank lines if any exist.
            if not line.strip():
                continue

            row = json.loads(line)

            # Every JSON object represents one utterance.
            total_records += 1

            # Duration is stored in seconds in the manifest.
            duration = float(row["duration"])
            total_duration += duration

            # Count an utterance once even if it contains multiple <unk> tags.
            if "<unk>" in row["text"]:
                unk_records += 1
                unk_duration += duration

    # What would remain if <unk> utterances were excluded.
    remaining_records = total_records - unk_records
    remaining_duration = total_duration - unk_duration

    return {
        "total_records": total_records,
        "unk_records": unk_records,
        "remaining_records": remaining_records,
        "total_duration": total_duration,
        "unk_duration": unk_duration,
        "remaining_duration": remaining_duration,
    }


def main():
    # Paths are supplied at runtime so this script is portable
    # across different hosts and container mount layouts.
    parser = argparse.ArgumentParser(
        description="Measure the impact of excluding <unk> utterances from ASR manifests."
    )

    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="One or more JSONL manifests to inspect",
    )

    args = parser.parse_args()

    # Produce one report per manifest.
    for manifest in args.manifests:
        result = inspect_manifest(manifest)

        print(f"\nManifest: {manifest}")
        print(f"Total records:       {result['total_records']}")
        print(f"Records with <unk>:  {result['unk_records']}")
        print(f"Records remaining:   {result['remaining_records']}")
        print(f"Total duration:      {result['total_duration'] / 3600:.2f} hours")
        print(f"<unk> duration:      {result['unk_duration'] / 3600:.2f} hours")
        print(f"Remaining duration:  {result['remaining_duration'] / 3600:.2f} hours")


if __name__ == "__main__":
    main()
