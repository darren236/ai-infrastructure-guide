import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Remove only these two explicitly approved annotation types.
NOISE_TAGS = ("<v-noise>", "<noise>")
TAG_PATTERN = re.compile(r"<[^<>]+>")


def normalize_manifest(source: Path, destination: Path) -> None:
    """Clean transcript annotations without changing the selected utterances."""
    records = []
    changed_records = 0
    removed_tags = Counter()

    # Validate and prepare every record before opening the output file.
    # Keeping this small POC manifest in memory is sufficient here.
    with source.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(
                    f"{source}:{line_number}: expected a record with string 'text'"
                )

            text = row["text"]

            # <unk> should already be absent. Stop on it or any other
            # unexpected tag rather than silently expanding our policy.
            unexpected = set(TAG_PATTERN.findall(text)) - set(NOISE_TAGS)
            if unexpected:
                raise ValueError(
                    f"{source}:{line_number}: unexpected tags: {sorted(unexpected)}"
                )

            cleaned = text
            for tag in NOISE_TAGS:
                removed_tags[tag] += text.count(tag)

                # Replace with a space so adjacent words do not join together.
                cleaned = cleaned.replace(tag, " ")

            # Collapse repeated whitespace and remove leading/trailing spaces.
            # Actual words, case, punctuation, and fillers are not rewritten.
            cleaned = " ".join(cleaned.split())

            # Do not silently drop an utterance and change our fixed subset.
            if not cleaned:
                raise ValueError(
                    f"{source}:{line_number}: transcript is empty after cleanup"
                )

            if cleaned != text:
                changed_records += 1

            # Only update text. Keep id, speaker, duration, audio, and any
            # other metadata unchanged. No audio file is opened or modified.
            row["text"] = cleaned
            records.append(row)

    if not records:
        raise ValueError(f"{source}: no records found")

    # Exclusive creation: fail if the destination already exists.
    # This prevents accidentally overwriting an input or previous output.
    with destination.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input:             {source}")
    print(f"Output:            {destination}")
    print(f"Records written:   {len(records)}")
    print(f"Text changed:      {changed_records}")
    for tag in NOISE_TAGS:
        print(f"Removed {tag}: {removed_tags[tag]}")


def main() -> None:
    # Paths are runtime arguments, not hardcoded host/container locations.
    parser = argparse.ArgumentParser(
        description="Remove approved noise tags from a POC JSONL manifest."
    )
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()

    try:
        normalize_manifest(args.input_manifest, args.output_manifest)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
