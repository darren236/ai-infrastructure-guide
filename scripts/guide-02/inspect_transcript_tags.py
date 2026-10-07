import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Match annotation-like tokens such as:
# <v-noise>
# <unk>
# <noise>
TAG_PATTERN = re.compile(r"<[^<>]+>")


def inspect_manifest(path):
    """
    Inspect transcript annotation tags in a JSONL ASR manifest.

    The source manifest is read-only.

    Returns:
      - total number of records
      - number of records containing tags
      - count of each tag found
    """
    tag_counts = Counter()
    records_with_tags = 0
    total_records = 0

    # Read the JSONL file one record at a time.
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            total_records += 1

            # Inspect only the transcript text.
            text = row["text"]
            tags = TAG_PATTERN.findall(text)

            # Count both affected records and individual tag occurrences.
            if tags:
                records_with_tags += 1
                tag_counts.update(tags)

    return total_records, records_with_tags, tag_counts


def main():
    # Accept one or more manifests so the same tool can inspect
    # train, validation, test, or other datasets.
    parser = argparse.ArgumentParser(
        description="Inspect transcript annotation tags in ASR manifests."
    )

    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="One or more JSONL manifests to inspect",
    )

    args = parser.parse_args()

    # Print a separate report for each supplied manifest.
    for manifest in args.manifests:
        total, tagged, counts = inspect_manifest(manifest)

        print(f"\nManifest: {manifest}")
        print(f"Total records:       {total}")
        print(f"Records with tags:   {tagged}")
        print("Tag counts:")

        if not counts:
            print("  none")
        else:
            for tag, count in counts.most_common():
                print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
