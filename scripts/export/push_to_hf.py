#!/usr/bin/env python3
"""
Push extracted audio snippets to Hugging Face dataset.

Usage:
    python push_to_hf.py <dataset_name>
    python push_to_hf.py <dataset_name> --input output/saved_snippets
    python push_to_hf.py user/dataset-name --private
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from datasets import Dataset, Audio
from tqdm import tqdm


def collect_snippets(input_dir: Path) -> List[Dict]:
    """Collect all snippets from input directory."""
    snippets = []

    # Find all JSON files
    json_files = sorted(input_dir.rglob('*.json'))

    for json_path in tqdm(json_files, desc="Collecting snippets"):
        # Find corresponding audio file
        audio_path = json_path.with_suffix('.mp3')

        if not audio_path.exists():
            print(f"Warning: Audio file not found for {json_path}", file=sys.stderr)
            continue

        # Load metadata
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}", file=sys.stderr)
            continue

        # Skip if transcription is empty
        transcription = metadata.get('transcription', '').strip()
        if not transcription:
            continue

        # Create dataset entry
        entry = {
            'audio': str(audio_path),
            'transcription': transcription,
            'language': metadata.get('language', ''),
            'quality_mos': metadata.get('quality_mos', 0.0),
        }

        snippets.append(entry)

    return snippets


def push_to_hf(
    dataset_name: str,
    input_dir: str = "output/saved_snippets",
    private: bool = False,
    sampling_rate: int = 24000,
):
    """Push snippets to Hugging Face dataset."""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"📂 Input directory: {input_dir}")
    print(f"🤗 Dataset name: {dataset_name}")
    print(f"🔒 Private: {private}")
    print(f"🎵 Sampling rate: {sampling_rate}Hz")

    # Collect all snippets
    print("\n📊 Collecting snippets...")
    snippets = collect_snippets(input_path)

    if not snippets:
        print("❌ No snippets found!", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Found {len(snippets):,} snippets")

    # Create dataset
    print("\n🚀 Creating dataset...")
    ds = Dataset.from_list(snippets)

    # Cast audio column
    print("🎵 Casting audio column...")
    ds = ds.cast_column('audio', Audio(sampling_rate=sampling_rate))

    # Shuffle dataset
    print("🔀 Shuffling dataset...")
    ds = ds.shuffle(seed=42)

    # Push to hub
    print(f"\n📤 Pushing to Hugging Face: {dataset_name}")
    ds.push_to_hub(dataset_name, private=private)

    print(f"\n✅ Dataset pushed successfully!")
    print(f"🔗 View at: https://huggingface.co/datasets/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Push extracted audio snippets to Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push to public dataset
  python push_to_hf.py user/dataset-name

  # Push to private dataset
  python push_to_hf.py user/dataset-name --private

  # Custom input directory
  python push_to_hf.py user/dataset-name --input data/snippets

  # Custom sampling rate
  python push_to_hf.py user/dataset-name --sampling-rate 16000

Note: You must be logged in to Hugging Face (huggingface-cli login)
        """,
    )

    parser.add_argument(
        "dataset_name",
        help="Hugging Face dataset name (e.g., user/dataset-name)",
    )

    parser.add_argument(
        "--input",
        "-i",
        default="output/saved_snippets",
        help="Input directory containing snippets (default: output/saved_snippets)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private",
    )

    parser.add_argument(
        "--sampling-rate",
        "-s",
        type=int,
        default=24000,
        help="Audio sampling rate (default: 24000)",
    )

    args = parser.parse_args()

    try:
        push_to_hf(
            args.dataset_name,
            input_dir=args.input,
            private=args.private,
            sampling_rate=args.sampling_rate,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
