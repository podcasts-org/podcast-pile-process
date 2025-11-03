#!/usr/bin/env python3
"""
Script to invalidate and mark for reprocessing all jobs processed after a specific date.

This script is useful when a bug in the worker processing code is discovered and
all jobs processed after a certain date need to be reprocessed with the fixed code.

Usage:
    # Dry run (default) - shows what would be invalidated without making changes
    python scripts/invalidate_recent_jobs.py --after "2025-10-29"

    # Actually invalidate the jobs
    python scripts/invalidate_recent_jobs.py --after "2025-10-29" --execute

    # Specify a datetime with time component
    python scripts/invalidate_recent_jobs.py --after "2025-10-29 15:30:00" --execute
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import podcastpile modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import func

from podcastpile.models import Job, JobStatus, get_db, init_db


def parse_datetime(date_string: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unable to parse date '{date_string}'. "
        f"Expected format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
    )


def invalidate_jobs(cutoff_date: datetime, dry_run: bool = True, batch_size: int = 1000) -> dict:
    """
    Invalidate jobs processed after the cutoff date using batched processing.

    Args:
        cutoff_date: Jobs processed after this date will be invalidated
        dry_run: If True, only show what would be done without making changes
        batch_size: Number of jobs to process per batch (default: 1000)

    Returns:
        Dictionary with statistics about the operation
    """
    # Initialize database
    init_db()

    # Get database session
    db = next(get_db())

    try:
        # First, get count of jobs to invalidate (indexed query)
        # We check both completed_at and processed_at to be thorough
        count_query = db.query(func.count(Job.id)).filter(
            Job.status == JobStatus.COMPLETED,
            (Job.processed_at > cutoff_date) |
            ((Job.processed_at == None) & (Job.completed_at > cutoff_date))
        )

        total_count = count_query.scalar()

        if total_count == 0:
            return {
                "total_found": 0,
                "invalidated": 0,
                "dry_run": dry_run,
                "jobs": []
            }

        print(f"Found {total_count} jobs to process...")

        # Query for job IDs only (much more memory efficient)
        # Use order_by for consistent pagination
        base_query = db.query(Job.id, Job.podcast_id, Job.episode_url,
                             Job.completed_at, Job.processed_at, Job.worker_id).filter(
            Job.status == JobStatus.COMPLETED,
            (Job.processed_at > cutoff_date) |
            ((Job.processed_at == None) & (Job.completed_at > cutoff_date))
        ).order_by(Job.id)

        # Collect lightweight job info for reporting
        job_info = []
        invalidated_count = 0

        # Process in batches
        offset = 0
        while True:
            # Fetch batch of job info
            batch = base_query.limit(batch_size).offset(offset).all()

            if not batch:
                break

            # Collect info for this batch
            batch_ids = []
            for job_data in batch:
                job_id, podcast_id, episode_url, completed_at, processed_at, worker_id = job_data

                info = {
                    "id": job_id,
                    "episode_url": episode_url,
                    "podcast_id": podcast_id,
                    "completed_at": completed_at.isoformat() if completed_at else None,
                    "processed_at": processed_at.isoformat() if processed_at else None,
                    "worker_id": worker_id,
                }
                job_info.append(info)
                batch_ids.append(job_id)

            if not dry_run:
                # Now fetch and update the actual Job objects for this batch
                jobs_to_update = db.query(Job).filter(Job.id.in_(batch_ids)).all()

                invalidation_timestamp = datetime.utcnow().isoformat()

                for job in jobs_to_update:
                    # Store original completion time before modifying
                    original_completion = job.completed_at

                    # Reset job to pending status
                    job.status = JobStatus.PENDING
                    job.worker_id = None
                    job.worker_ip = None
                    job.assigned_at = None
                    job.completed_at = None
                    job.expires_at = None

                    # Clear results but document in error_message for reference
                    old_error = job.error_message or ""
                    invalidation_note = (
                        f"[INVALIDATED {invalidation_timestamp}] "
                        f"Job reprocessed due to worker bug. "
                        f"Original completion: {original_completion}. "
                    )
                    if old_error:
                        job.error_message = invalidation_note + "\nPrevious error: " + old_error
                    else:
                        job.error_message = invalidation_note

                    # Clear processing results
                    job.transcription = None
                    job.diarization = None
                    job.result_json = None
                    job.processing_duration = None
                    job.worker_gpu = None
                    job.processed_at = None

                    # Increment retry count to track this was reprocessed
                    job.retry_count = (job.retry_count or 0) + 1

                # Commit this batch
                db.commit()
                invalidated_count += len(batch_ids)

                # Progress reporting
                print(f"  Invalidated {invalidated_count:,} / {total_count:,} jobs...")

            offset += batch_size

            # Small optimization: if we're in dry-run and we've collected enough info, break
            if dry_run and offset >= min(10000, total_count):
                # Continue collecting metadata but don't load all jobs in dry-run
                remaining = total_count - offset
                if remaining > 0:
                    print(f"  [Dry run: Scanned {offset:,} jobs, skipping detailed scan of remaining {remaining:,}]")
                break

        if not dry_run:
            print(f"✓ Successfully invalidated {invalidated_count:,} jobs")

        return {
            "total_found": total_count,
            "invalidated": invalidated_count if not dry_run else 0,
            "dry_run": dry_run,
            "jobs": job_info
        }

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Invalidate jobs processed after a specific date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Invalidate jobs processed after this date (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the invalidation (default is dry-run)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about each job"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of jobs to process per batch (default: 1000)"
    )

    args = parser.parse_args()

    # Parse the cutoff date
    try:
        cutoff_date = parse_datetime(args.after)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the invalidation
    print(f"{'DRY RUN - ' if not args.execute else ''}Searching for jobs processed after {cutoff_date}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 80)

    try:
        result = invalidate_jobs(cutoff_date, dry_run=not args.execute, batch_size=args.batch_size)

        print(f"\nFound {result['total_found']} jobs processed after {cutoff_date}")

        if result['total_found'] > 0:
            # Show summary statistics
            jobs = result['jobs']

            # Group by podcast_id
            podcast_counts = {}
            for job in jobs:
                pid = job['podcast_id'] or 'Unknown'
                podcast_counts[pid] = podcast_counts.get(pid, 0) + 1

            print(f"\nBreakdown by podcast:")
            for podcast_id, count in sorted(podcast_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {podcast_id}: {count} jobs")

            # Show sample jobs if verbose
            if args.verbose:
                print(f"\nJob details:")
                for job in jobs[:50]:  # Limit to first 50 for readability
                    print(f"  Job #{job['id']}: {job['episode_url']}")
                    print(f"    Podcast: {job['podcast_id']}")
                    print(f"    Completed: {job['completed_at']}")
                    print(f"    Processed: {job['processed_at']}")
                    print(f"    Worker: {job['worker_id']}")
                    print()

                if len(jobs) > 50:
                    print(f"  ... and {len(jobs) - 50} more jobs")

            if result['dry_run']:
                print(f"\n{'='*80}")
                print(f"This was a DRY RUN - no changes were made")
                print(f"Run with --execute to actually invalidate these jobs")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"Successfully invalidated {result['invalidated']} jobs")
                print(f"These jobs are now back in PENDING status and will be reprocessed")
                print(f"{'='*80}")
        else:
            print(f"\nNo jobs found to invalidate")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
