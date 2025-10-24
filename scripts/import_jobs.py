#!/usr/bin/env python3
"""
Fast bulk import of jobs from JSONL file into the database.

Usage:
    python scripts/import_jobs.py input.jsonl [--batch-size 1000] [--skip-existing]

For 2M+ rows, this script uses:
- Batch inserts (default 1000 rows per batch)
- Direct SQL with bulk insert
- Skip duplicate checking via temp table
- Progress reporting every 10k rows
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Iterator, List, Dict

# Add parent directory to path to import podcastpile
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from podcastpile.models.database import Base
from podcastpile.models.job import Job, JobStatus
from podcastpile.config import config


def parse_language(lang_str: str) -> str:
    """Extract first 2 letters of language code in lowercase"""
    if not lang_str:
        return None
    # Take first 2 characters before any dash/underscore
    lang = lang_str.split('-')[0].split('_')[0]
    return lang[:2].lower()


def read_jsonl_batched(filepath: str, batch_size: int) -> Iterator[List[Dict]]:
    """Read JSONL file in batches"""
    batch = []
    total_read = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                batch.append(data)
                total_read += 1

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue

    # Yield remaining items
    if batch:
        yield batch


def get_existing_urls(session, urls: List[str]) -> set:
    """Get set of URLs that already exist in database"""
    if not urls:
        return set()

    # SQLAlchemy requires creating individual placeholders for IN clause
    # For large batches, check in chunks to avoid SQL length limits
    chunk_size = 500
    existing = set()

    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]

        # Create placeholders: :url0, :url1, :url2, etc.
        placeholders = ', '.join(f':url{j}' for j in range(len(chunk)))
        query = text(f"SELECT episode_url FROM jobs WHERE episode_url IN ({placeholders})")

        # Create parameter dict: {url0: 'http://...', url1: 'http://...', ...}
        params = {f'url{j}': url for j, url in enumerate(chunk)}

        result = session.execute(query, params)
        existing.update(row[0] for row in result)

    return existing


def bulk_insert_jobs(session, jobs_data: List[Dict], skip_existing: bool = True) -> tuple:
    """
    Bulk insert jobs using SQLAlchemy core for maximum speed.

    Returns:
        (inserted_count, skipped_count)
    """
    if not jobs_data:
        return 0, 0

    # Prepare job records
    records = []
    urls_to_check = []

    for item in jobs_data:
        episode_url = item.get('episode_url')
        if not episode_url:
            continue

        # Parse language
        language = parse_language(item.get('language'))
        podcast_id = item.get('podcast_id')

        records.append({
            'episode_url': episode_url,
            'podcast_id': podcast_id,
            'language': language,
            'status': JobStatus.PENDING.value,
            'worker_id': None,
            'worker_ip': None,
            'transcription': None,
            'diarization': None,
            'result_json': None,
            'processing_duration': None,
            'worker_gpu': None,
            'processed_at': None,
            'assigned_at': None,
            'completed_at': None,
            'expires_at': None,
            'error_message': None,
            'retry_count': 0,
            # created_at will use default (utcnow)
        })
        urls_to_check.append(episode_url)

    if not records:
        return 0, 0

    # Check for existing URLs if skip_existing is True
    existing_urls = set()
    if skip_existing:
        existing_urls = get_existing_urls(session, urls_to_check)

    # Filter out duplicates
    new_records = [r for r in records if r['episode_url'] not in existing_urls]
    skipped = len(records) - len(new_records)

    if not new_records:
        return 0, skipped

    # Bulk insert using Core insert
    try:
        session.execute(
            Job.__table__.insert(),
            new_records
        )
        session.commit()
        return len(new_records), skipped

    except Exception as e:
        session.rollback()
        print(f"Error during bulk insert: {e}", file=sys.stderr)
        raise


def import_jsonl(
    filepath: str,
    batch_size: int = 1000,
    skip_existing: bool = True,
    progress_interval: int = 10000
):
    """
    Import JSONL file into database with batching and progress reporting.

    Args:
        filepath: Path to JSONL file
        batch_size: Number of records per batch insert
        skip_existing: Skip URLs that already exist
        progress_interval: Report progress every N records
    """
    # Setup database connection with optimizations for SQLite
    connect_args = {}
    if 'sqlite' in config.DATABASE_URL:
        # SQLite performance optimizations
        connect_args = {
            'check_same_thread': False,
        }

    engine = create_engine(
        config.DATABASE_URL,
        echo=False,
        connect_args=connect_args
    )

    # Apply SQLite-specific pragmas for massive speed boost
    if 'sqlite' in config.DATABASE_URL:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode = WAL"))  # Write-Ahead Logging
            conn.execute(text("PRAGMA synchronous = NORMAL"))  # Faster than FULL
            conn.execute(text("PRAGMA cache_size = -64000"))  # 64MB cache
            conn.execute(text("PRAGMA temp_store = MEMORY"))  # Use memory for temp
            conn.commit()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Stats
    total_processed = 0
    total_inserted = 0
    total_skipped = 0
    start_time = time.time()

    print(f"Starting import from {filepath}")
    print(f"Batch size: {batch_size}")
    print(f"Skip existing: {skip_existing}")
    if not skip_existing:
        print("⚡ FAST MODE: Duplicate checking disabled")
    print("-" * 60)

    try:
        session = Session()

        # For fresh imports, use single transaction for max speed
        if not skip_existing and 'sqlite' in config.DATABASE_URL:
            session.execute(text("BEGIN"))

        for batch in read_jsonl_batched(filepath, batch_size):
            inserted, skipped = bulk_insert_jobs(session, batch, skip_existing)

            total_processed += len(batch)
            total_inserted += inserted
            total_skipped += skipped

            # Progress reporting
            if total_processed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"Processed: {total_processed:,} | "
                      f"Inserted: {total_inserted:,} | "
                      f"Skipped: {total_skipped:,} | "
                      f"Rate: {rate:.0f} rows/sec")

        session.close()

    except KeyboardInterrupt:
        print("\n\nImport interrupted by user")
        session.rollback()
        session.close()

    except Exception as e:
        print(f"\n\nError during import: {e}", file=sys.stderr)
        session.rollback()
        session.close()
        raise

    # Final stats
    elapsed = time.time() - start_time
    rate = total_processed / elapsed if elapsed > 0 else 0

    print("-" * 60)
    print(f"Import complete!")
    print(f"  Total processed: {total_processed:,}")
    print(f"  Inserted: {total_inserted:,}")
    print(f"  Skipped (duplicates): {total_skipped:,}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Average rate: {rate:.0f} rows/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Fast bulk import of jobs from JSONL file"
    )
    parser.add_argument(
        "input_file",
        help="Path to JSONL file with job data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records per batch insert (default: 1000)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing URLs (may cause errors on duplicates)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10000,
        help="Report progress every N records (default: 10000)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    import_jsonl(
        args.input_file,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        progress_interval=args.progress_interval
    )


if __name__ == "__main__":
    main()
