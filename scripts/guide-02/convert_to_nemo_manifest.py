import argparse
import json
from pathlib import Path


def convert_manifest(input_path, audio_root, output_path, language):
    """
    Convert our normalized NSC manifest into Nemotron-compatible NeMo format.

    Input fields:
        id, speaker, duration, text, audio

    Output fields:
        audio_filepath, duration, text, lang, target_lang

    The original manifest and audio files remain unchanged.
    """

    # Resolve the source audio directory inside the container.
    # Example: /data/nsc/nsc_query_5h
    audio_root = audio_root.resolve()

    converted = []

    # Read the normalized JSONL manifest.
    # Each line represents one utterance.
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            # Skip empty lines.
            if not line.strip():
                continue

            row = json.loads(line)

            # NSC stores relative audio paths.
            # Example: audio/example.flac
            relative_audio = Path(row["audio"])

            # We expect a relative path, not an absolute one.
            if relative_audio.is_absolute():
                raise ValueError(
                    f"Line {line_number}: expected a relative audio path"
                )

            # Construct the absolute audio path INSIDE the container.
            #
            # Example:
            # audio_root    = /data/nsc/nsc_query_5h
            # relative_audio = audio/example.flac
            #
            # Result:
            # /data/nsc/nsc_query_5h/audio/example.flac
            audio_path = (audio_root / relative_audio).resolve()

            # Prevent malformed paths from escaping the source directory.
            if not audio_path.is_relative_to(audio_root):
                raise ValueError(
                    f"Line {line_number}: audio path escapes source directory"
                )

            # Verify that the audio file actually exists.
            # This checks file existence, not audio decoding.
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Line {line_number}: {audio_path}"
                )

            # Create the five fields used by NVIDIA's
            # Nemotron prompt-aware ASR fine-tuning recipe.
            nemo_record = {
                "audio_filepath": str(audio_path),
                "duration": float(row["duration"]),
                "text": row["text"],
                "lang": language,
                "target_lang": language,
            }

            converted.append(nemo_record)

    # Prevent accidentally creating an empty training manifest.
    if not converted:
        raise ValueError("No records found in input manifest")

    # Create a new output file.
    # Mode "x" prevents overwriting an existing file.
    with output_path.open("x", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print a summary for operational verification.
    print(f"Input:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Records written: {len(converted)}")
    print(f"Language:        {language}")
    print("Audio paths:     verified")


def main():
    # Define command-line arguments.
    # This keeps the script independent of Brev-specific paths.
    parser = argparse.ArgumentParser(
        description="Convert normalized NSC manifests to Nemotron ASR format."
    )

    # Path to the normalized input manifest.
    parser.add_argument("input_manifest", type=Path)

    # Directory containing the original audio/ subdirectory.
    parser.add_argument("audio_root", type=Path)

    # Destination for the new NeMo manifest.
    parser.add_argument("output_manifest", type=Path)

    # Language conditioning for the Nemotron model.
    # en-US is our chosen supported English locale for this POC.
    parser.add_argument(
        "--language",
        default="en-US",
        help="Nemotron language locale (default: en-US)",
    )

    args = parser.parse_args()

    # Execute the conversion using the supplied arguments.
    convert_manifest(
        args.input_manifest,
        args.audio_root,
        args.output_manifest,
        args.language,
    )


# Run main() only when executed directly.
if __name__ == "__main__":
    main()
