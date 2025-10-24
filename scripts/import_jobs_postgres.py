#!/usr/bin/env python3
"""
ULTRA-FAST bulk import for PostgreSQL using COPY command.

This is 10-100x faster than regular inserts for PostgreSQL.
For 2M rows, this can complete in seconds instead of minutes.

Usage:
    python scripts/import_jobs_postgres.py input.jsonl

Requires PostgreSQL database (won't work with SQLite).
"""

import sys
import json
import argparse
import time
import tempfile
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text
from podcastpile.config import config


def parse_language(lang_str: str) -> str:
    """Extract first 2 letters of language code in lowercase"""
    if not lang_str:
        return ''
    lang = lang_str.split('-')[0].split('_')[0]
    return lang[:2].lower()


def import_jsonl_postgres(filepath: str):
    """
    Ultra-fast import using PostgreSQL COPY command.

    Strategy:
    1. Load all data into a temporary table
    2. Use INSERT INTO ... SELECT with NOT EXISTS to skip duplicates
    3. All done in a single transaction for maximum speed
    """
    print(f"Starting PostgreSQL COPY import from {filepath}")
    print("=" * 60)

    if 'postgresql' not in config.DATABASE_URL and 'postgres' not in config.DATABASE_URL:
        print("ERROR: This script requires PostgreSQL.", file=sys.stderr)
        print("For SQLite, use import_jobs.py instead", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(config.DATABASE_URL, echo=False)
    start_time = time.time()

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
        writer = csv.writer(tmp)

        # Write header (must match temp table columns)
        writer.writerow(['episode_url', 'podcast_id', 'language'])

        # Convert JSONL to CSV
        total_rows = 0
        print("Converting JSONL to CSV...")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    episode_url = data.get('episode_url')
                    if not episode_url:
                        continue

                    podcast_id = data.get('podcast_id', '')
                    language = parse_language(data.get('language'))

                    writer.writerow([episode_url, podcast_id, language])
                    total_rows += 1

                    if total_rows % 100000 == 0:
                        print(f"  Converted {total_rows:,} rows...")

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                    continue

    convert_time = time.time() - start_time
    print(f"✓ Converted {total_rows:,} rows in {convert_time:.2f}s")
    print()

    # Import into PostgreSQL
    print("Importing into PostgreSQL...")

    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()

            # Create temporary table
            print("  Creating temporary table...")
            conn.execute(text("""
                CREATE TEMPORARY TABLE temp_jobs (
                    episode_url VARCHAR NOT NULL,
                    podcast_id VARCHAR,
                    language VARCHAR
                )
            """))

            # COPY data from CSV into temp table
            print("  Loading data with COPY...")
            copy_start = time.time()

            # Use raw connection for COPY
            raw_conn = conn.connection
            cursor = raw_conn.cursor()

            with open(tmp_path, 'r') as f:
                # Skip header
                next(f)
                cursor.copy_expert(
                    """
                    COPY temp_jobs (episode_url, podcast_id, language)
                    FROM STDIN WITH CSV
                    """,
                    f
                )

            copy_time = time.time() - copy_start
            print(f"  ✓ COPY completed in {copy_time:.2f}s ({total_rows/copy_time:.0f} rows/sec)")

            # Insert from temp table, skipping duplicates
            print("  Inserting non-duplicate rows...")
            insert_start = time.time()

            result = conn.execute(text("""
                INSERT INTO jobs (
                    episode_url,
                    podcast_id,
                    language,
                    status,
                    worker_id,
                    worker_ip,
                    transcription,
                    diarization,
                    result_json,
                    processing_duration,
                    worker_gpu,
                    processed_at,
                    assigned_at,
                    completed_at,
                    expires_at,
                    error_message,
                    retry_count,
                    created_at
                )
                SELECT
                    t.episode_url,
                    NULLIF(t.podcast_id, ''),
                    t.language,
                    'pending',
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    0,
                    NOW()
                FROM temp_jobs t
                WHERE NOT EXISTS (
                    SELECT 1 FROM jobs j WHERE j.episode_url = t.episode_url
                )
            """))

            inserted = result.rowcount
            insert_time = time.time() - insert_start
            print(f"  ✓ Inserted {inserted:,} rows in {insert_time:.2f}s")

            # Commit transaction
            trans.commit()

            # Cleanup temp file
            Path(tmp_path).unlink()

            # Final stats
            total_time = time.time() - start_time
            skipped = total_rows - inserted

            print()
            print("=" * 60)
            print("Import complete!")
            print(f"  Total rows processed: {total_rows:,}")
            print(f"  Inserted: {inserted:,}")
            print(f"  Skipped (duplicates): {skipped:,}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average rate: {total_rows/total_time:.0f} rows/sec")
            print(f"  Insert rate: {inserted/insert_time:.0f} rows/sec")

    except Exception as e:
        print(f"\nError during import: {e}", file=sys.stderr)
        # Cleanup temp file
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-fast PostgreSQL bulk import using COPY"
    )
    parser.add_argument(
        "input_file",
        help="Path to JSONL file with job data"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    import_jsonl_postgres(args.input_file)


if __name__ == "__main__":
    main()
