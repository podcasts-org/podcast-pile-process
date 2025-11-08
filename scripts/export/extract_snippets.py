#!/usr/bin/env python3
"""
Extract high-quality audio snippets from downloaded podcast files - ULTRA-FAST version.

Optimizations:
- Parallel ffmpeg extraction with joblib for multi-core utilization
- Batch processing to minimize I/O overhead
- Memory-efficient streaming processing
- Smart folder bucketing to avoid filesystem limits
- orjson for fast JSON parsing

Performance: Processes hundreds of snippets per second on multi-core systems.

Usage:
    python extract_snippets.py
    python extract_snippets.py --min-mos 4.5
    python extract_snippets.py --input output/audio_files --output output/saved_snippets
    python extract_snippets.py --min-mos 4.8 --parallel 16
    python extract_snippets.py --snippets-per-folder 5000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json as orjson
    USE_ORJSON = False

from joblib import Parallel, delayed
from tqdm import tqdm


def load_json(json_path: Path) -> Optional[Dict]:
    """Load JSON file with orjson if available."""
    try:
        with open(json_path, 'rb') as f:
            if USE_ORJSON:
                return orjson.loads(f.read())
            else:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}", file=sys.stderr)
        return None


def save_json(data: Dict, json_path: Path):
    """Save JSON file with orjson if available."""
    with open(json_path, 'wb' if USE_ORJSON else 'w') as f:
        if USE_ORJSON:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        else:
            json.dump(data, f, indent=2)


def extract_snippet(
    audio_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> bool:
    """
    Extract audio snippet using ffmpeg with stream copy and metadata stripping (ULTRA-FAST).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temp file for atomic operation
        temp_path = output_path.with_suffix('.tmp.mp3')

        # ffmpeg command with precise seeking AND metadata stripping
        # -ss BEFORE -i for fast seek (keyframe), -ss AFTER -i for precise seek
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Fast seek to approximate position
            '-i', str(audio_path),
            '-t', str(duration),  # Duration to extract
            '-map_metadata', '-1',  # Strip ALL metadata (cover art, tags, etc.)
            '-map', '0:a',  # Copy only audio stream
            '-c', 'copy',  # Stream copy - NO re-encoding (SPEED!)
            '-y',
            str(temp_path)
        ]

        # Run ffmpeg (suppress output)
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,  # 60s timeout per snippet
        )

        if result.returncode == 0 and temp_path.exists():
            # Atomic rename
            temp_path.rename(output_path)
            return True
        else:
            # Cleanup failed attempt
            if temp_path.exists():
                temp_path.unlink()
            return False

    except Exception as e:
        print(f"Error extracting snippet {output_path}: {e}", file=sys.stderr)
        return False


def process_audio_file(
    audio_path: Path,
    json_path: Path,
    output_dir: Path,
    min_mos: float,
    snippets_per_folder: int,
    global_snippet_counter: int,
) -> Tuple[int, List[Dict]]:
    """
    Process a single audio file and extract high-quality snippets.

    Returns:
        Tuple of (num_snippets_extracted, snippet_metadata_list)
    """
    # Load JSON metadata
    metadata = load_json(json_path)
    if not metadata:
        return (0, [])

    # Get SHA256 of original file
    file_hashes = metadata.get('file_hashes', {})
    sha256 = file_hashes.get('sha256', 'unknown')

    # Get language from top level
    language = metadata.get('language', '')

    # Get segments
    segments = metadata.get('segments', [])

    # Filter high-quality segments
    high_quality_segments = [
        seg for seg in segments
        if seg.get('quality_mos') is not None and seg.get('quality_mos') >= min_mos
    ]

    if not high_quality_segments:
        return (0, [])

    # Extract each segment
    extracted_count = 0
    snippet_metadata_list = []

    for segment in high_quality_segments:
        snippet_num = global_snippet_counter + extracted_count

        # Determine folder (0000/, 0001/, etc.)
        folder_num = snippet_num // snippets_per_folder
        folder_name = f"{folder_num:04d}"
        snippet_folder = output_dir / folder_name
        snippet_folder.mkdir(parents=True, exist_ok=True)

        # Determine snippet filename (1.mp3, 2.mp3, etc.)
        local_snippet_num = snippet_num % snippets_per_folder + 1
        snippet_audio_path = snippet_folder / f"{local_snippet_num}.mp3"
        snippet_json_path = snippet_folder / f"{local_snippet_num}.json"

        # Extract segment
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time

        if duration <= 0:
            continue

        # Extract audio snippet
        success = extract_snippet(
            audio_path,
            snippet_audio_path,
            start_time,
            duration,
        )

        if not success:
            continue

        # Create snippet metadata
        snippet_metadata = {
            'original_sha256': sha256,
            'original_file': audio_path.name,
            'segment_index': segment.get('index', -1),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'quality_mos': segment.get('quality_mos'),
            'transcription': segment.get('transcription', ''),
            'language': language,
            'speaker': segment.get('speaker', ''),
        }

        # Save snippet metadata
        save_json(snippet_metadata, snippet_json_path)

        extracted_count += 1
        snippet_metadata_list.append(snippet_metadata)

    return (extracted_count, snippet_metadata_list)


def collect_audio_files(input_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Collect all audio files and their corresponding JSON metadata.

    Returns:
        List of (audio_path, json_path) tuples
    """
    audio_files = []

    # Scan all subdirectories
    for json_path in input_dir.rglob('*.json'):
        # Find corresponding audio file
        audio_path = json_path.with_suffix('.mp3')

        if audio_path.exists():
            audio_files.append((audio_path, json_path))

    return audio_files


def extract_snippets(
    input_dir: str,
    output_dir: str,
    min_mos: float = 4.8,
    snippets_per_folder: int = 1000,
    n_jobs: int = -1,
):
    """
    Extract high-quality audio snippets from downloaded podcast files.

    Args:
        input_dir: Directory containing audio files and JSON metadata
        output_dir: Directory to save extracted snippets
        min_mos: Minimum MOS score for snippet extraction
        snippets_per_folder: Number of snippets per folder (for bucketing)
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    start_time = time.time()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Minimum MOS: {min_mos}")
    print(f"📁 Snippets per folder: {snippets_per_folder}")

    if USE_ORJSON:
        print("⚡ Using orjson for fast JSON parsing")

    if n_jobs == -1:
        import multiprocessing
        actual_jobs = multiprocessing.cpu_count()
        print(f"🔥 Using {actual_jobs} CPU cores for parallel processing")
    elif n_jobs > 1:
        print(f"🔥 Using {n_jobs} parallel jobs")

    # Collect all audio files
    print("\n📊 Scanning for audio files...")
    audio_files = collect_audio_files(input_path)

    if not audio_files:
        print("❌ No audio files found!")
        sys.exit(1)

    print(f"✅ Found {len(audio_files):,} audio files")

    # First pass: Analyze segments and collect stats
    print("\n📊 Analyzing segments for quality filtering...")

    all_segments = []
    total_files = 0
    total_segments = 0
    high_quality_segments = 0
    durations = []

    for audio_path, json_path in tqdm(audio_files, desc="Analyzing"):
        metadata = load_json(json_path)
        if not metadata:
            continue

        total_files += 1
        segments = metadata.get('segments', [])
        total_segments += len(segments)

        for segment in segments:
            mos = segment.get('quality_mos')
            if mos is not None and mos >= min_mos:
                high_quality_segments += 1
                duration = segment.get('end', 0) - segment.get('start', 0)
                if duration > 0:
                    durations.append(duration)
                    all_segments.append((audio_path, json_path, segment))

    # Print statistics
    print(f"\n🎯 Quality Stats (MOS ≥ {min_mos}):")
    print(f"   Total files: {total_files:,}")
    print(f"   Total segments: {total_segments:,}")
    print(f"   High-quality segments (≥{min_mos}): {high_quality_segments:,}")
    print(f"   Files with high-quality segments: {len(set(ap for ap, _, _ in all_segments)):,}")

    if durations:
        total_duration = sum(durations)
        print(f"\n📊 Duration Stats (high-quality segments):")
        print(f"   Total duration: {total_duration:,.1f}s ({total_duration/3600:.1f} hours)")
        print(f"   Average: {statistics.mean(durations):.2f}s")
        print(f"   Median: {statistics.median(durations):.2f}s")
        print(f"   Min: {min(durations):.2f}s")
        print(f"   Max: {max(durations):.2f}s")
        if len(durations) > 1:
            print(f"   Std dev: {statistics.stdev(durations):.2f}s")

    if not all_segments:
        print("\n❌ No high-quality segments found!")
        sys.exit(0)

    # Group segments by file for efficient processing
    print(f"\n🚀 Extracting {high_quality_segments:,} snippets...")

    file_segments = {}
    for audio_path, json_path, segment in all_segments:
        key = (audio_path, json_path)
        if key not in file_segments:
            file_segments[key] = []
        file_segments[key].append(segment)

    # Process files with segments
    tasks = []
    global_counter = 0

    for (audio_path, json_path), segments in file_segments.items():
        # Load full metadata for this file
        metadata = load_json(json_path)
        if not metadata:
            continue

        # Get language from metadata
        language = metadata.get('language', '')

        # Create task for each segment
        for segment in segments:
            tasks.append((
                audio_path,
                json_path,
                segment,
                metadata.get('file_hashes', {}).get('sha256', 'unknown'),
                language,
                global_counter,
            ))
            global_counter += 1

    # Process in parallel
    def extract_single_snippet(task):
        audio_path, json_path, segment, sha256, language, snippet_num = task

        # Determine folder
        folder_num = snippet_num // snippets_per_folder
        folder_name = f"{folder_num:04d}"
        snippet_folder = output_path / folder_name
        snippet_folder.mkdir(parents=True, exist_ok=True)

        # Determine snippet filename
        local_snippet_num = snippet_num % snippets_per_folder + 1
        snippet_audio_path = snippet_folder / f"{local_snippet_num}.mp3"
        snippet_json_path = snippet_folder / f"{local_snippet_num}.json"

        # Extract segment
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time

        if duration <= 0:
            return False

        # Extract audio snippet
        success = extract_snippet(
            audio_path,
            snippet_audio_path,
            start_time,
            duration,
        )

        if not success:
            return False

        # Create snippet metadata
        snippet_metadata = {
            'original_sha256': sha256,
            'original_file': audio_path.name,
            'segment_index': segment.get('index', -1),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'quality_mos': segment.get('quality_mos'),
            'transcription': segment.get('transcription', ''),
            'language': language,
            'speaker': segment.get('speaker', ''),
        }

        # Save snippet metadata
        save_json(snippet_metadata, snippet_json_path)

        return True

    # Run parallel extraction
    if n_jobs == 1:
        results = [extract_single_snippet(task) for task in tqdm(tasks, desc="Extracting")]
    else:
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(extract_single_snippet)(task)
            for task in tqdm(tasks, desc="Extracting")
        )

    # Count successes
    successful = sum(1 for r in results if r)

    elapsed = time.time() - start_time

    # Print final statistics
    print(f"\n✅ Extraction complete!")
    print(f"   Extracted: {successful:,} / {len(tasks):,} snippets")
    print(f"   Failed: {len(tasks) - successful:,}")
    print(f"   Processing time: {elapsed:.2f}s")
    print(f"   Throughput: {successful/elapsed:,.0f} snippets/s")
    print(f"\n📂 Snippets saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality audio snippets from downloaded podcast files (ULTRA-FAST)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with default settings (MOS ≥ 4.8)
  python extract_snippets.py

  # Extract with lower MOS threshold
  python extract_snippets.py --min-mos 4.5

  # Custom input/output directories
  python extract_snippets.py --input data/audio --output data/snippets

  # Use specific number of parallel workers
  python extract_snippets.py --parallel 16

  # Custom folder bucketing
  python extract_snippets.py --snippets-per-folder 5000

Performance tips:
  - Install orjson for faster JSON parsing: pip install orjson
  - Use all CPU cores (default) for maximum speed
  - Ensure ffmpeg is installed and in PATH
  - Use SSD storage for best I/O performance
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        default="output/audio_files",
        help="Input directory containing audio files and JSON metadata (default: output/audio_files)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output/saved_snippets",
        help="Output directory for extracted snippets (default: output/saved_snippets)",
    )

    parser.add_argument(
        "--min-mos",
        "-m",
        type=float,
        default=4.8,
        help="Minimum MOS score for snippet extraction (default: 4.8)",
    )

    parser.add_argument(
        "--snippets-per-folder",
        "-s",
        type=int,
        default=1000,
        help="Number of snippets per folder for bucketing (default: 1000)",
    )

    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores, default: -1)",
    )

    args = parser.parse_args()

    # Validate MOS range
    if not (0 <= args.min_mos <= 5):
        print("Error: --min-mos must be between 0 and 5", file=sys.stderr)
        sys.exit(1)

    # Warn if orjson not available
    if not USE_ORJSON:
        print("Warning: orjson not installed, using slower standard json", file=sys.stderr)
        print("Install with: pip install orjson\n", file=sys.stderr)

    try:
        extract_snippets(
            args.input,
            args.output,
            min_mos=args.min_mos,
            snippets_per_folder=args.snippets_per_folder,
            n_jobs=args.parallel,
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
