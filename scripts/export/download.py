#!/usr/bin/env python3
"""
Download all processed podcasts from S3 - parallel downloader.

Optimizations:
- Parallel downloads with joblib
- Organized folder structure (0000/, 0001/, etc with 1000 files each)
- Smart resume (skip already downloaded)
- Connection pooling
- Progress tracking with tqdm
- Metadata extraction and cleaning

Performance: Downloads 10k+ files in minutes with parallel workers.

Output structure:
  output/audio_files/
    0000/
      {hash}.mp3
      {hash}.json
    0001/
      {hash}.mp3
      {hash}.json
    ...

Usage:
    python download.py
    python download.py --json podcasts.jsonl --hashes s3_urls.json
    python download.py --parallel 50 --resume
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from joblib import Parallel, delayed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

# Check if ffmpeg is available
FFMPEG_AVAILABLE = False
try:
    result = subprocess.run(['ffmpeg', '-version'],
                          capture_output=True,
                          timeout=1)
    FFMPEG_AVAILABLE = result.returncode == 0
except:
    pass


class S3Downloader:
    """Ultra-fast parallel S3 downloader."""

    def __init__(self, s3_config: Optional[Dict] = None, max_pool_connections: int = 100):
        """
        Initialize S3 downloader with connection pooling.

        Args:
            s3_config: S3 configuration dict
            max_pool_connections: Max connections for pool
        """
        if s3_config is None:
            s3_config = self._load_config_from_env()

        self.bucket = s3_config["bucket"]
        self.endpoint_url = s3_config["endpoint_url"]

        # Configure connection pooling for parallel downloads
        botocore_config = BotocoreConfig(
            max_pool_connections=max_pool_connections,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            tcp_keepalive=True,
        )

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=s3_config["endpoint_url"],
            aws_access_key_id=s3_config["access_key_id"],
            aws_secret_access_key=s3_config["secret_access_key"],
            config=botocore_config,
            region_name=s3_config.get("region", "us-east-1"),
        )

    @staticmethod
    def _load_config_from_env() -> Dict:
        """Load S3 configuration from environment variables."""
        required_vars = [
            "S3_ENDPOINT_URL",
            "S3_ACCESS_KEY_ID",
            "S3_SECRET_ACCESS_KEY",
            "S3_BUCKET",
        ]

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please set: S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET"
            )

        return {
            "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
            "access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
            "bucket": os.getenv("S3_BUCKET"),
            "region": os.getenv("S3_REGION", "us-east-1"),
        }

    @retry(
        stop=stop_after_attempt(3),  # 3 attempts total
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s
        retry=retry_if_exception_type((ClientError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a single file from S3 with AUTOMATIC RETRIES and partial download tracking.

        Retries: 3 attempts with exponential backoff (2s, 4s, 8s)
        Retries on: S3 errors, connection errors, timeouts

        Args:
            s3_key: S3 object key
            local_path: Local file path

        Returns:
            True if successful

        Raises:
            Exception if download fails after all retries
        """
        # Create parent directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Use temporary file during download
        import uuid
        temp_suffix = f".{uuid.uuid4().hex[:8]}"
        temp_path = local_path + temp_suffix

        try:
            # Download to temp file
            self.s3_client.download_file(self.bucket, s3_key, temp_path)

            # Atomic rename on success
            os.rename(temp_path, local_path)
            return True
        except:
            # Clean up partial download
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise


def get_folder_number(index: int, files_per_folder: int = 1000) -> str:
    """
    Get folder number for given index.

    Args:
        index: File index
        files_per_folder: Number of files per folder

    Returns:
        Folder name (e.g., "0000", "0001")
    """
    folder_num = index // files_per_folder
    return f"{folder_num:04d}"


def clean_result_json(result_json_str: str) -> Dict:
    """
    Parse and clean result_json, removing specified keys.

    Args:
        result_json_str: JSON string from database

    Returns:
        Cleaned dict
    """
    # Parse JSON
    if USE_ORJSON:
        result = orjson.loads(result_json_str)
    else:
        result = json.loads(result_json_str)

    # Remove keys
    keys_to_remove = ['processed_at', 'gpu_info', 'bgm_model', 'worker_version', 'audio_filepath']
    for key in keys_to_remove:
        result.pop(key, None)

    return result


def load_jsonl(file_path: str) -> Dict[str, Dict]:
    """
    Load JSONL file and create hash -> metadata mapping.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dict mapping SHA256 hash -> full record
    """
    is_compressed = file_path.endswith('.gz')

    if is_compressed:
        f = gzip.open(file_path, 'rt', encoding='utf-8')
    else:
        f = open(file_path, 'r', encoding='utf-8')

    hash_to_record = {}

    try:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if USE_ORJSON:
                record = orjson.loads(line)
            else:
                record = json.loads(line)

            # Extract hash from result_json
            result_json_str = record.get('result_json')
            if not result_json_str:
                continue

            try:
                if USE_ORJSON:
                    result_data = orjson.loads(result_json_str)
                else:
                    result_data = json.loads(result_json_str)

                file_hashes = result_data.get('file_hashes', {})
                sha256 = file_hashes.get('sha256')

                if sha256:
                    hash_to_record[sha256] = record
            except:
                continue

    finally:
        f.close()

    return hash_to_record


def load_s3_mapping(file_path: str) -> Dict[str, Dict]:
    """
    Load S3 URL mapping JSON.

    Args:
        file_path: Path to JSON file

    Returns:
        Dict mapping SHA256 hash -> S3 info
    """
    with open(file_path, 'r') as f:
        if USE_ORJSON:
            return orjson.loads(f.read())
        else:
            return json.load(f)


def download_single_file(
    args: Tuple[str, Dict, Dict, str, int, bool]
) -> Tuple[bool, str, Optional[str]]:
    """
    Download a single file and its metadata (for parallel processing).

    Args:
        args: Tuple of (hash, s3_info, record, output_dir, index, resume)

    Returns:
        Tuple of (success, hash, error_message)
    """
    sha256_hash, s3_info, record, output_dir, index, resume = args

    # Get folder number
    folder = get_folder_number(index)
    folder_path = os.path.join(output_dir, folder)

    # Determine file extension from S3 key
    s3_key = s3_info.get('key')
    if not s3_key:
        return (False, sha256_hash, "Missing S3 key in mapping")

    extension = '.mp3'  # default
    if '.' in s3_key:
        extension = '.' + s3_key.rsplit('.', 1)[1]

    audio_path = os.path.join(folder_path, f"{sha256_hash}{extension}")
    json_path = os.path.join(folder_path, f"{sha256_hash}.json")

    # Skip if resume and already exists
    if resume and os.path.exists(audio_path) and os.path.exists(json_path):
        return (True, sha256_hash, None)

    # Create folder
    os.makedirs(folder_path, exist_ok=True)

    # Initialize S3 client (one per worker)
    try:
        s3_config = {
            "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
            "access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
            "bucket": os.getenv("S3_BUCKET"),
            "region": os.getenv("S3_REGION", "us-east-1"),
        }

        downloader = S3Downloader(s3_config, max_pool_connections=10)

        # Download audio file
        if not os.path.exists(audio_path):
            try:
                success = downloader.download_file(s3_key, audio_path)
                if not success:
                    return (False, sha256_hash, f"S3 download failed for key: {s3_key}")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return (False, sha256_hash, f"S3 error ({error_code}): {s3_key}")
            except Exception as e:
                return (False, sha256_hash, f"Download error: {str(e)}")

        # Write metadata JSON
        if not os.path.exists(json_path):
            result_json_str = record.get('result_json')
            if result_json_str:
                try:
                    cleaned_data = clean_result_json(result_json_str)

                    with open(json_path, 'w') as f:
                        if USE_ORJSON:
                            f.write(orjson.dumps(cleaned_data, option=orjson.OPT_INDENT_2).decode())
                        else:
                            json.dump(cleaned_data, f, indent=2)
                except Exception as e:
                    return (False, sha256_hash, f"Failed to write JSON: {e}")

        return (True, sha256_hash, None)

    except Exception as e:
        return (False, sha256_hash, f"Unexpected error: {str(e)}")


def strip_metadata_fast(audio_path: str) -> bool:
    """
    BLAZING-FAST metadata stripping using ffmpeg.

    Strips ALL metadata, cover art, ID3 tags, etc. from audio files.
    Uses ffmpeg with copy codec (no re-encoding) for MAXIMUM SPEED.

    Args:
        audio_path: Path to audio file

    Returns:
        True if successful
    """
    if not FFMPEG_AVAILABLE:
        return False

    import uuid
    temp_suffix = f".strip_{uuid.uuid4().hex[:8]}"
    temp_path = audio_path + temp_suffix

    try:
        # ffmpeg with stream copy (no re-encode) - BLAZING FAST
        # -map_metadata -1: Strip ALL metadata
        # -c copy: Copy streams without re-encoding (SPEED!)
        # -y: Overwrite temp file
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-map_metadata', '-1',  # Strip metadata
            '-map', '0:a',  # Copy audio stream
            '-c', 'copy',  # No re-encoding (SPEED!)
            '-y',  # Overwrite
            '-loglevel', 'error',  # Quiet
            temp_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,  # 30s timeout per file
            check=False
        )

        if result.returncode == 0 and os.path.exists(temp_path):
            # Atomic replace
            os.replace(temp_path, audio_path)
            return True
        else:
            # Cleanup failed temp
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def strip_metadata_batch(
    audio_files: list,
    n_jobs: int = -1,
    show_progress: bool = True
) -> Tuple[int, int]:
    """
    ULTRA-FAST parallel metadata stripping.

    Args:
        audio_files: List of audio file paths
        n_jobs: Number of parallel jobs
        show_progress: Show progress bar

    Returns:
        Tuple of (success_count, fail_count)
    """
    if not FFMPEG_AVAILABLE:
        print("⚠️  ffmpeg not found - skipping metadata stripping")
        print("   Install with: brew install ffmpeg  (or apt/yum)")
        return (0, 0)

    print(f"\n🔥 Stripping metadata from {len(audio_files):,} files...")

    if n_jobs == -1:
        import multiprocessing
        actual_jobs = multiprocessing.cpu_count() * 2  # 2x for CPU-bound
        print(f"   Using {actual_jobs} parallel workers")
    else:
        actual_jobs = n_jobs

    # Strip metadata in parallel
    results = Parallel(n_jobs=actual_jobs, backend='threading', verbose=0)(
        delayed(strip_metadata_fast)(path)
        for path in tqdm(
            audio_files,
            desc="Stripping metadata",
            disable=not show_progress,
            unit="file"
        )
    )

    success = sum(1 for r in results if r)
    failed = len(results) - success

    print(f"✓ Stripped metadata: {success:,} files")
    if failed > 0:
        print(f"⚠️  Failed: {failed:,} files")

    return (success, failed)


def cleanup_partial_downloads(output_dir: str):
    """
    Clean up partial download files (files with temp suffixes).

    Args:
        output_dir: Output directory to scan
    """
    print("Cleaning up partial downloads...")
    count = 0

    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            # Match patterns:
            # - Download temps: filename.{8 hex chars}
            # - Strip temps: filename.strip_{8 hex chars}
            if len(filename) > 9 and filename[-9] == '.':
                suffix = filename[-8:]
                if all(c in '0123456789abcdef' for c in suffix):
                    filepath = os.path.join(root, filename)
                    try:
                        os.remove(filepath)
                        count += 1
                    except:
                        pass
            elif '.strip_' in filename:
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    count += 1
                except:
                    pass

    if count > 0:
        print(f"✓ Removed {count} partial file(s)")


def has_high_quality_segments(record: Dict, min_mos: float = 4.5) -> bool:
    """
    Check if ANY segment has MOS >= min_mos.

    Args:
        record: Job record with result_json
        min_mos: Minimum MOS score threshold

    Returns:
        True if any segment meets threshold
    """
    result_json_str = record.get('result_json')
    if not result_json_str:
        return False

    try:
        if USE_ORJSON:
            result_data = orjson.loads(result_json_str)
        else:
            result_data = json.loads(result_json_str)

        segments = result_data.get('segments', [])

        # Check if ANY segment has MOS >= min_mos
        # Field is 'quality_mos' not 'mos'
        for segment in segments:
            mos = segment.get('quality_mos')
            if mos is not None and mos >= min_mos:
                return True

        return False
    except:
        return False


def download_all(
    jsonl_path: str,
    s3_mapping_path: str,
    output_dir: str,
    n_jobs: int = -1,
    resume: bool = True,
    strip_metadata: bool = True,
    min_mos: Optional[float] = None,
):
    """
    Download all files in parallel - MAXIMUM THROUGHPUT MODE.

    Args:
        jsonl_path: Path to processed_podcasts.jsonl
        s3_mapping_path: Path to S3 mapping JSON (hash -> S3 URL)
        output_dir: Output directory
        n_jobs: Number of parallel jobs (-1 for auto, >0 for custom, e.g. 200 for SATURATION)
        resume: Skip already downloaded files
        strip_metadata: Strip metadata/cover art from audio files (BLAZING FAST)
        min_mos: Only download if ANY segment has MOS >= this value (e.g. 4.5)
    """
    start_time = time.time()

    # Clean up any partial downloads from previous runs
    if os.path.exists(output_dir):
        cleanup_partial_downloads(output_dir)

    print(f"Loading metadata from: {jsonl_path}")
    hash_to_record = load_jsonl(jsonl_path)
    print(f"✓ Loaded {len(hash_to_record):,} records")

    print(f"\nLoading S3 mapping from: {s3_mapping_path}")
    s3_mapping = load_s3_mapping(s3_mapping_path)
    print(f"✓ Loaded {len(s3_mapping):,} S3 URLs")

    # Filter to only hashes with S3 info and collect MOS stats
    download_tasks = []
    filtered_by_mos = 0
    total_high_quality_segments = 0  # Count segments with MOS >= threshold

    for i, (sha256_hash, s3_info) in enumerate(s3_mapping.items()):
        if s3_info is None:
            continue

        record = hash_to_record.get(sha256_hash)
        if not record:
            continue

        # Count high-quality segments regardless of filter
        if min_mos is not None:
            result_json_str = record.get('result_json')
            if result_json_str:
                try:
                    if USE_ORJSON:
                        result_data = orjson.loads(result_json_str)
                    else:
                        result_data = json.loads(result_json_str)

                    segments = result_data.get('segments', [])
                    for segment in segments:
                        mos = segment.get('quality_mos')
                        if mos is not None and mos >= min_mos:
                            total_high_quality_segments += 1
                except:
                    pass

        # Filter by MOS if requested
        if min_mos is not None:
            if not has_high_quality_segments(record, min_mos):
                filtered_by_mos += 1
                continue

        download_tasks.append((sha256_hash, s3_info, record, output_dir, i, resume))

    if min_mos is not None:
        print(f"\n🎯 MOS Quality Stats (≥{min_mos}):")
        print(f"   Total segments with MOS ≥{min_mos}: {total_high_quality_segments:,}")
        print(f"   Files with at least 1 segment ≥{min_mos}: {len(download_tasks):,}")
        print(f"   Files filtered out: {filtered_by_mos:,}")
        print(f"   Average high-quality segments per file: {total_high_quality_segments / len(download_tasks):.1f}" if download_tasks else "   No files to download")

    print(f"\nPrepared {len(download_tasks):,} download tasks")

    if n_jobs == -1:
        import multiprocessing
        # Auto mode: 4x CPU cores for I/O-bound task (downloads are network I/O)
        actual_jobs = multiprocessing.cpu_count() * 4
        print(f"Using {actual_jobs} parallel workers (4x {multiprocessing.cpu_count()} cores - AUTO SATURATION MODE)")
    elif n_jobs > 1:
        print(f"Using {n_jobs} parallel workers (SATURATION MODE 🔥)")
    else:
        print("Using sequential mode")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download in parallel with threading (best for I/O-bound downloads)
    print(f"\nDownloading to: {output_dir}")
    print("Folder structure: 0000/ (1000 files), 0001/ (1000 files), ...\n")

    try:
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(download_single_file)(task)
            for task in tqdm(download_tasks, desc="Downloading", unit="file")
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted! Cleaning up partial downloads...")
        cleanup_partial_downloads(output_dir)
        print("✓ Cleanup complete")
        raise

    # Count results
    successful = sum(1 for success, _, _ in results if success)
    failed = len(results) - successful

    download_time = time.time() - start_time

    print(f"\n✓ Download complete!")
    print(f"  Successful: {successful:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Time: {download_time:.1f}s")
    print(f"  Throughput: {successful/download_time:.1f} files/s")

    # Strip metadata if requested
    if strip_metadata and successful > 0:
        # Collect all downloaded audio files
        audio_files = []
        for success, hash_val, error in results:
            if success:
                # Find the audio file for this hash
                for i, (sha256_hash, s3_info, record, _, _, _) in enumerate(download_tasks):
                    if sha256_hash == hash_val:
                        folder = get_folder_number(i)
                        folder_path = os.path.join(output_dir, folder)

                        # Get extension from S3 key
                        s3_key = s3_info.get('key', '')
                        extension = '.mp3'
                        if '.' in s3_key:
                            extension = '.' + s3_key.rsplit('.', 1)[1]

                        audio_path = os.path.join(folder_path, f"{sha256_hash}{extension}")
                        if os.path.exists(audio_path):
                            audio_files.append(audio_path)
                        break

        if audio_files:
            strip_metadata_batch(audio_files, n_jobs=n_jobs)

    elapsed = time.time() - start_time

    print(f"\n🎉 Total time: {elapsed:.1f}s")

    # Show failed files
    if failed > 0:
        print(f"\nFailed files:")
        for success, hash_val, error in results:
            if not success:
                print(f"  {hash_val[:16]}... - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all processed podcasts from S3 (ULTRA-FAST parallel mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with AUTO SATURATION (4x CPU cores)
  python download.py

  # MAXIMUM SATURATION (200 parallel workers) 🔥💪
  python download.py --parallel 200

  # Ultra saturation (500+ workers if your S3 can handle it)
  python download.py --parallel 500

  # ONLY download high-quality audio (MOS >= 4.5)
  python download.py --min-mos 4.5

  # ULTRA-HIGH quality only (MOS >= 4.8)
  python download.py --min-mos 4.8 --parallel 200

  # Conservative mode (50 workers)
  python download.py --parallel 50

  # Custom paths
  python download.py --json podcasts.jsonl --hashes s3_urls.json

  # Sequential mode (slower but lower memory)
  python download.py --parallel 1

  # No resume (re-download everything)
  python download.py --no-resume

Output structure:
  output/audio_files/
    0000/
      a1b2c3...mp3
      a1b2c3...json
      b2c3d4...mp3
      b2c3d4...json
      (1000 files total)
    0001/
      (next 1000 files)
    ...

Performance tips:
  - Install orjson for faster JSON parsing: pip install orjson
  - Default auto mode uses 4x CPU cores (best for most cases)
  - Use --parallel 200-500 for MAXIMUM SATURATION 💪
  - Downloads are I/O-bound, so 100+ workers is totally fine
  - Resume is enabled by default (skips existing files)
  - Partial downloads auto-cleanup on Ctrl+C
  - Each folder contains 1000 files for filesystem efficiency
  - Automatic retries: 3 attempts with exponential backoff (2s, 4s, 8s)
  - Metadata stripping: BLAZING-FAST ffmpeg -c copy (no re-encoding)

Saturation guide:
  - Local S3 (fast network): --parallel 200-500
  - Cloud S3 (AWS): --parallel 100-200
  - Slow network: --parallel 50
  - Your S3 can handle it 🌶️

Resilience:
  - Auto-retry on network errors, timeouts, S3 throttling
  - Exponential backoff prevents S3 hammering
  - Partial downloads cleaned up automatically
  - Resume on script restart (skips completed files)
        """,
    )

    parser.add_argument(
        "--json",
        default="processed_podcasts.jsonl",
        help="Path to processed_podcasts.jsonl (default: processed_podcasts.jsonl)",
    )

    parser.add_argument(
        "--hashes",
        default="hashes.json",
        help="Path to S3 mapping JSON (default: hashes.json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output/audio_files",
        help="Output directory (default: output/audio_files)",
    )

    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores, default: -1)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip existing files (re-download everything)",
    )

    parser.add_argument(
        "--no-strip-metadata",
        action="store_true",
        help="Don't strip metadata/cover art from audio files",
    )

    parser.add_argument(
        "--min-mos",
        type=float,
        help="Only download files where ANY segment has MOS >= this value (e.g., 4.5 for high quality)",
    )

    parser.add_argument(
        "--files-per-folder",
        type=int,
        default=1000,
        help="Files per folder (default: 1000)",
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.json).exists():
        print(f"Error: File not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.hashes).exists():
        print(f"Error: File not found: {args.hashes}", file=sys.stderr)
        sys.exit(1)

    # Warn if orjson not available
    if not USE_ORJSON:
        print("Warning: orjson not installed, using slower standard json")
        print("Install with: pip install orjson\n")

    # Warn if ffmpeg not available and metadata stripping is requested
    if not FFMPEG_AVAILABLE and not args.no_strip_metadata:
        print("Warning: ffmpeg not found - metadata stripping will be skipped")
        print("Install with: brew install ffmpeg  (or apt/yum)\n")

    try:
        download_all(
            args.json,
            args.hashes,
            args.output,
            n_jobs=args.parallel,
            resume=not args.no_resume,
            strip_metadata=not args.no_strip_metadata,
            min_mos=args.min_mos,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
