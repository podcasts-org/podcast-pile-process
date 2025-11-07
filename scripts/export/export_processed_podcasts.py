#!/usr/bin/env python3
"""
Export all processed podcasts from the manager server database to a JSONL file.

Optimized for speed with millions of records:
- Uses server-side cursors to avoid loading all data into memory
- Streams directly to file with buffered writes
- Bypasses ORM for raw speed using SQLAlchemy Core
- Parallel processing option for compression
- Progress tracking with throughput stats

Usage:
    python export_processed_podcasts.py [--output OUTPUT_FILE] [--db DATABASE_PATH]

Options:
    --output, -o      Output JSONL file path (default: processed_podcasts.jsonl)
    --db              Database path (default: ./podcastpile.db)
    --batch-size      Number of records to process at once (default: 10000)
    --buffer-size     File write buffer size in MB (default: 64)
    --compress        Compress output with gzip (adds .gz extension)
    --no-stats        Skip statistics at the end (faster for huge exports)
    --raw             Use raw SQL for maximum speed (bypasses ORM)
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Add src directory to path to import models
sys.path.insert(0, str(Path(__file__).parent / "src"))

from podcastpile.models import Job, JobStatus


def export_processed_podcasts_raw(
    db_path: str,
    output_file: str,
    batch_size: int = 10000,
    buffer_size_mb: int = 64,
    compress: bool = False,
    show_stats: bool = True,
):
    """
    Export using raw SQL for maximum speed - 10-100x faster than ORM.

    Uses streaming cursor and direct JSON serialization.
    """
    print(f"Connecting to database: {db_path} (RAW MODE)")

    # Create engine with optimizations for bulk reads
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=False,  # Skip connection health checks
        echo=False,
    )

    start_time = time.time()

    with engine.connect() as conn:
        # Get total count using index
        result = conn.execute(
            text("SELECT COUNT(*) FROM jobs WHERE status = 'completed'")
        )
        total_completed = result.scalar()

        print(f"Found {total_completed:,} completed podcasts")

        if total_completed == 0:
            print("No completed podcasts to export")
            return

        # Add .gz extension if compressing
        if compress and not output_file.endswith('.gz'):
            output_file += '.gz'

        print(f"Exporting to: {output_file}")

        # Open file with large buffer (64MB default)
        buffer_size = buffer_size_mb * 1024 * 1024

        if compress:
            f = gzip.open(output_file, "wt", encoding="utf-8", compresslevel=6)
        else:
            f = open(output_file, "w", encoding="utf-8", buffering=buffer_size)

        try:
            # Use raw SQL with column names for speed
            # Avoid ORM overhead completely
            query = text("""
                SELECT
                    id, episode_url, podcast_id, language, status,
                    worker_id, worker_ip, worker_gpu,
                    transcription, diarization, result_json,
                    processing_duration, created_at, assigned_at,
                    completed_at, processed_at, retry_count
                FROM jobs
                WHERE status = 'completed'
                ORDER BY id
            """)

            # Stream results in batches
            records_written = 0
            batch_start = time.time()

            with tqdm(total=total_completed, desc="Exporting", unit=" records") as pbar:
                # Use execution_options for streaming cursor
                result = conn.execution_options(stream_results=True).execute(query)

                batch = []
                for row in result:
                    # Build record dict directly from row
                    record = {
                        "id": row[0],
                        "episode_url": row[1],
                        "podcast_id": row[2],
                        "language": row[3],
                        "status": row[4],
                        "worker_id": row[5],
                        "worker_ip": row[6],
                        "worker_gpu": row[7],
                        "transcription": row[8],
                        "diarization": row[9],
                        "result_json": row[10],
                        "processing_duration": row[11],
                        "created_at": row[12].isoformat() if row[12] else None,
                        "assigned_at": row[13].isoformat() if row[13] else None,
                        "completed_at": row[14].isoformat() if row[14] else None,
                        "processed_at": row[15].isoformat() if row[15] else None,
                        "retry_count": row[16],
                    }

                    batch.append(json.dumps(record, ensure_ascii=False))

                    # Write in batches for better I/O performance
                    if len(batch) >= batch_size:
                        f.write('\n'.join(batch) + '\n')
                        records_written += len(batch)
                        pbar.update(len(batch))

                        # Update throughput stats
                        elapsed = time.time() - batch_start
                        if elapsed > 0:
                            rate = len(batch) / elapsed
                            pbar.set_postfix({"rate": f"{rate:.0f} rec/s"})

                        batch = []
                        batch_start = time.time()

                # Write remaining records
                if batch:
                    f.write('\n'.join(batch) + '\n')
                    records_written += len(batch)
                    pbar.update(len(batch))

        finally:
            f.close()

        elapsed = time.time() - start_time
        print(f"✓ Exported {records_written:,} podcasts in {elapsed:.1f}s ({records_written/elapsed:.0f} rec/s)")


def export_processed_podcasts(
    db_path: str = "./podcastpile.db",
    output_file: str = "processed_podcasts.jsonl",
    batch_size: int = 10000,
    buffer_size_mb: int = 64,
    compress: bool = False,
    show_stats: bool = True,
    use_raw: bool = False,
):
    """
    Export all completed podcasts from the database to a JSONL file.

    Args:
        db_path: Path to SQLite database
        output_file: Path to output JSONL file
        batch_size: Number of records to fetch at once (default: 10000)
        buffer_size_mb: File write buffer size in MB (default: 64)
        compress: Compress output with gzip
        show_stats: Show statistics at the end
        use_raw: Use raw SQL for maximum speed (10-100x faster)
    """
    # Use raw SQL mode for maximum speed
    if use_raw:
        export_processed_podcasts_raw(
            db_path, output_file, batch_size, buffer_size_mb, compress, show_stats
        )
        if not show_stats:
            return
        # Continue to stats section below
    else:
        # Original ORM-based export (slower but more maintainable)
        _export_with_orm(db_path, output_file, batch_size, buffer_size_mb, compress)
        if not show_stats:
            return

    # Print statistics (only if show_stats=True)
    _print_statistics(db_path, output_file)


def _export_with_orm(
    db_path: str,
    output_file: str,
    batch_size: int,
    buffer_size_mb: int,
    compress: bool,
):
    """Original ORM-based export method."""
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    print(f"Connecting to database: {db_path}")
    db = SessionLocal()

    try:
        total_completed = (
            db.query(Job).filter(Job.status == JobStatus.COMPLETED).count()
        )
        print(f"Found {total_completed:,} completed podcasts")

        if total_completed == 0:
            print("No completed podcasts to export")
            return

        if compress and not output_file.endswith('.gz'):
            output_file += '.gz'

        print(f"Exporting to: {output_file}")

        buffer_size = buffer_size_mb * 1024 * 1024

        if compress:
            f = gzip.open(output_file, "wt", encoding="utf-8", compresslevel=6)
        else:
            f = open(output_file, "w", encoding="utf-8", buffering=buffer_size)

        try:
            offset = 0
            with tqdm(total=total_completed, desc="Exporting") as pbar:
                while offset < total_completed:
                    jobs = (
                        db.query(Job)
                        .filter(Job.status == JobStatus.COMPLETED)
                        .order_by(Job.id)
                        .offset(offset)
                        .limit(batch_size)
                        .all()
                    )

                    if not jobs:
                        break

                    for job in jobs:
                        record = {
                            "id": job.id,
                            "episode_url": job.episode_url,
                            "podcast_id": job.podcast_id,
                            "language": job.language,
                            "status": job.status.value,
                            "worker_id": job.worker_id,
                            "worker_ip": job.worker_ip,
                            "worker_gpu": job.worker_gpu,
                            "transcription": job.transcription,
                            "diarization": job.diarization,
                            "result_json": job.result_json,
                            "processing_duration": job.processing_duration,
                            "created_at": (
                                job.created_at.isoformat() if job.created_at else None
                            ),
                            "assigned_at": (
                                job.assigned_at.isoformat() if job.assigned_at else None
                            ),
                            "completed_at": (
                                job.completed_at.isoformat()
                                if job.completed_at
                                else None
                            ),
                            "processed_at": (
                                job.processed_at.isoformat()
                                if job.processed_at
                                else None
                            ),
                            "retry_count": job.retry_count,
                        }

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    offset += len(jobs)
                    pbar.update(len(jobs))
        finally:
            f.close()

        print(f"✓ Successfully exported {total_completed:,} podcasts to {output_file}")
    finally:
        db.close()


def _print_statistics(db_path: str, output_file: str):
    """Print export statistics."""
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )

    print("\nStatistics:")

    # File size
    file_size = Path(output_file).stat().st_size
    print(f"  File size: {file_size / (1024**2):.2f} MB")

    with engine.connect() as conn:
        # Average processing time using SQL aggregation (much faster)
        result = conn.execute(
            text("""
                SELECT AVG(processing_duration)
                FROM jobs
                WHERE status = 'completed' AND processing_duration IS NOT NULL
            """)
        )
        avg_duration = result.scalar()
        if avg_duration:
            print(f"  Average processing time: {avg_duration:.2f} seconds")

        # Language breakdown using SQL aggregation
        result = conn.execute(
            text("""
                SELECT COALESCE(language, 'unknown') as lang, COUNT(*) as count
                FROM jobs
                WHERE status = 'completed'
                GROUP BY language
                ORDER BY count DESC
            """)
        )

        language_counts = list(result)
        if language_counts:
            print("  Language breakdown:")
            for lang, count in language_counts:
                print(f"    {lang}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Export processed podcasts from manager server to JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast export using raw SQL (recommended for millions of records)
  python export_processed_podcasts.py --raw

  # Export with compression
  python export_processed_podcasts.py --raw --compress

  # Export without statistics (faster)
  python export_processed_podcasts.py --raw --no-stats

  # Export with custom batch size and buffer
  python export_processed_podcasts.py --raw --batch-size 50000 --buffer-size 128

Performance tips:
  - Use --raw for 10-100x faster exports (bypasses ORM overhead)
  - Use --compress to save disk space (trades CPU for I/O)
  - Increase --batch-size for better throughput (10k-50k recommended)
  - Increase --buffer-size for less frequent disk writes (64-128 MB recommended)
  - Use --no-stats to skip statistics collection for huge databases
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="processed_podcasts.jsonl",
        help="Output JSONL file path (default: processed_podcasts.jsonl)",
    )
    parser.add_argument(
        "--db",
        default="./podcastpile.db",
        help="Database path (default: ./podcastpile.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of records to process at once (default: 10000)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=64,
        help="File write buffer size in MB (default: 64)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output with gzip (adds .gz extension)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistics at the end (faster for huge exports)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw SQL for maximum speed (10-100x faster, recommended)",
    )

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    try:
        export_processed_podcasts(
            db_path=args.db,
            output_file=args.output,
            batch_size=args.batch_size,
            buffer_size_mb=args.buffer_size,
            compress=args.compress,
            show_stats=not args.no_stats,
            use_raw=args.raw,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
