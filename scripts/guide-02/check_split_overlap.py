import argparse
import json
from pathlib import Path


def load_manifest(path):
    """
    Read a JSONL manifest into memory.

    Each non-empty line must contain one JSON object.
    This function only reads the source file.
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Define the command-line interface.
    # Manifest paths are supplied at runtime so this script is not tied
    # to a specific host, user account, or dataset directory.
    parser = argparse.ArgumentParser(
        description="Check speaker and utterance-ID overlap between two ASR manifests."
    )

    parser.add_argument(
        "train_manifest",
        type=Path,
        help="Path to the training JSONL manifest",
    )

    parser.add_argument(
        "dev_manifest",
        type=Path,
        help="Path to the validation JSONL manifest",
    )

    args = parser.parse_args()

    # Load both manifests from paths supplied at runtime.
    train = load_manifest(args.train_manifest)
    dev = load_manifest(args.dev_manifest)

    # Collect unique speaker IDs.
    train_speakers = {row["speaker"] for row in train}
    dev_speakers = {row["speaker"] for row in dev}

    # Collect unique utterance IDs.
    train_ids = {row["id"] for row in train}
    dev_ids = {row["id"] for row in dev}

    # Set intersection (&) gives values present in both datasets.
    speaker_overlap = train_speakers & dev_speakers
    id_overlap = train_ids & dev_ids

    # Print a concise integrity report.
    print(f"Train records:   {len(train)}")
    print(f"Dev records:     {len(dev)}")
    print(f"Train speakers:  {len(train_speakers)}")
    print(f"Dev speakers:    {len(dev_speakers)}")
    print(f"Speaker overlap: {len(speaker_overlap)}")
    print(f"ID overlap:      {len(id_overlap)}")


if __name__ == "__main__":
    main()
