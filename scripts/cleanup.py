#!/usr/bin/env python3
"""
Script to clean up stuck jobs that have been processing for too long.

This script finds jobs that have been in PROCESSING or ASSIGNED status for longer
than the specified timeout and resets them to PENDING so they can be picked up again.

This is useful for:
- Worker crashes that didn't report failures
- Network issues causing workers to disconnect
- Jobs that got stuck for any reason

Usage:
    # Dry run (default) - shows what would be cleaned up
    python scripts/cleanup.py

    # Actually clean up stuck jobs
    python scripts/cleanup.py --execute

    # Custom timeout (default is 2 hours)
    python scripts/cleanup.py --timeout 4 --execute

    # Clean up only specific statuses
    python scripts/cleanup.py --status PROCESSING --execute
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import podcastpile modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import func, or_

from podcastpile.models import Job, JobStatus, get_db, init_db


def cleanup_stuck_jobs(
    timeout_hours: float = 2.0,
    dry_run: bool = True,
    batch_size: int = 1000,
    statuses: list = None
) -> dict:
    """
    Clean up jobs that have been stuck in processing/assigned status.

    Args:
        timeout_hours: Reset jobs older than this many hours (default: 2.0)
        dry_run: If True, only show what would be done without making changes
        batch_size: Number of jobs to process per batch (default: 1000)
        statuses: List of JobStatus to check (default: PROCESSING and ASSIGNED)

    Returns:
        Dictionary with statistics about the operation
    """
    # Initialize database
    init_db()

    # Get database session
    db = next(get_db())

    # Default to checking both PROCESSING and ASSIGNED
    if statuses is None:
        statuses = [JobStatus.PROCESSING, JobStatus.ASSIGNED]

    try:
        # Calculate cutoff time
        cutoff_time = datetime.utcnow() - timedelta(hours=timeout_hours)

        # Build filter conditions
        # Jobs are stuck if:
        # 1. Status is PROCESSING or ASSIGNED
        # 2. assigned_at is older than cutoff time
        # 3. OR expires_at has passed (jobs with expires_at set that are overdue)
        filter_conditions = [
            Job.status.in_(statuses),
            or_(
                Job.assigned_at < cutoff_time,
                (Job.expires_at != None) & (Job.expires_at < datetime.utcnow())
            )
        ]

        # First, get count of jobs to clean up (indexed query)
        count_query = db.query(func.count(Job.id)).filter(*filter_conditions)
        total_count = count_query.scalar()

        if total_count == 0:
            return {
                "total_found": 0,
                "cleaned": 0,
                "dry_run": dry_run,
                "jobs": []
            }

        print(f"Found {total_count:,} stuck jobs to clean up...")

        # Query for job info (memory efficient - only essential fields)
        # Use order_by for consistent pagination
        base_query = db.query(
            Job.id, Job.status, Job.worker_id, Job.episode_url,
            Job.podcast_id, Job.assigned_at, Job.expires_at
        ).filter(*filter_conditions).order_by(Job.id)

        # Collect lightweight job info for reporting
        job_info = []
        cleaned_count = 0

        # Statistics by status and worker
        status_counts = {}
        worker_counts = {}

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
                job_id, status, worker_id, episode_url, podcast_id, assigned_at, expires_at = job_data

                info = {
                    "id": job_id,
                    "status": status.value,
                    "worker_id": worker_id,
                    "episode_url": episode_url,
                    "podcast_id": podcast_id,
                    "assigned_at": assigned_at.isoformat() if assigned_at else None,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "stuck_duration_hours": (
                        (datetime.utcnow() - assigned_at).total_seconds() / 3600
                        if assigned_at else None
                    )
                }
                job_info.append(info)
                batch_ids.append(job_id)

                # Update statistics
                status_key = status.value
                status_counts[status_key] = status_counts.get(status_key, 0) + 1

                worker_key = worker_id or "Unknown"
                worker_counts[worker_key] = worker_counts.get(worker_key, 0) + 1

            if not dry_run:
                # Now fetch and update the actual Job objects for this batch
                jobs_to_update = db.query(Job).filter(Job.id.in_(batch_ids)).all()

                cleanup_timestamp = datetime.utcnow().isoformat()

                for job in jobs_to_update:
                    # Store original state for error message
                    original_status = job.status.value
                    original_worker = job.worker_id
                    original_assigned = job.assigned_at

                    # Reset job to pending status
                    job.status = JobStatus.PENDING
                    job.worker_id = None
                    job.worker_ip = None
                    job.assigned_at = None
                    job.expires_at = None

                    # Document the cleanup in error_message for tracking
                    old_error = job.error_message or ""
                    cleanup_note = (
                        f"[CLEANED UP {cleanup_timestamp}] "
                        f"Job was stuck in {original_status} status for too long. "
                        f"Worker: {original_worker}, "
                        f"Assigned: {original_assigned}. "
                    )
                    if old_error:
                        job.error_message = cleanup_note + "\nPrevious error: " + old_error
                    else:
                        job.error_message = cleanup_note

                    # Increment retry count to track this was retried
                    job.retry_count = (job.retry_count or 0) + 1

                # Commit this batch
                db.commit()
                cleaned_count += len(batch_ids)

                # Progress reporting
                print(f"  Cleaned up {cleaned_count:,} / {total_count:,} jobs...")

            offset += batch_size

            # Optimization: in dry-run, limit detailed scanning
            if dry_run and offset >= min(10000, total_count):
                remaining = total_count - offset
                if remaining > 0:
                    print(f"  [Dry run: Scanned {offset:,} jobs, skipping detailed scan of remaining {remaining:,}]")
                break

        if not dry_run:
            print(f"✓ Successfully cleaned up {cleaned_count:,} stuck jobs")

        return {
            "total_found": total_count,
            "cleaned": cleaned_count if not dry_run else 0,
            "dry_run": dry_run,
            "jobs": job_info,
            "status_counts": status_counts,
            "worker_counts": worker_counts,
            "cutoff_time": cutoff_time.isoformat()
        }

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Clean up stuck jobs that have been processing too long",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Reset jobs older than this many hours (default: 2.0)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the cleanup (default is dry-run)"
    )
    parser.add_argument(
        "--status",
        choices=["PROCESSING", "ASSIGNED", "BOTH"],
        default="BOTH",
        help="Which status to clean up (default: BOTH)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of jobs to process per batch (default: 1000)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about each job"
    )

    args = parser.parse_args()

    # Determine which statuses to check
    if args.status == "PROCESSING":
        statuses = [JobStatus.PROCESSING]
    elif args.status == "ASSIGNED":
        statuses = [JobStatus.ASSIGNED]
    else:
        statuses = [JobStatus.PROCESSING, JobStatus.ASSIGNED]

    # Run the cleanup
    print(f"{'DRY RUN - ' if not args.execute else ''}Cleaning up stuck jobs")
    print(f"Timeout: {args.timeout} hours")
    print(f"Checking statuses: {', '.join([s.value for s in statuses])}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 80)

    try:
        result = cleanup_stuck_jobs(
            timeout_hours=args.timeout,
            dry_run=not args.execute,
            batch_size=args.batch_size,
            statuses=statuses
        )

        if result['total_found'] == 0:
            print(f"\n✓ No stuck jobs found - everything is running smoothly!")
        else:
            print(f"\nFound {result['total_found']:,} stuck jobs")
            print(f"Cutoff time: Jobs assigned before {result['cutoff_time']}")

            # Show breakdown by status
            if result['status_counts']:
                print(f"\nBreakdown by status:")
                for status, count in sorted(result['status_counts'].items()):
                    print(f"  {status}: {count:,} jobs")

            # Show breakdown by worker
            if result['worker_counts']:
                print(f"\nBreakdown by worker:")
                for worker, count in sorted(
                    result['worker_counts'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]:  # Top 20 workers
                    print(f"  {worker}: {count:,} jobs")

                if len(result['worker_counts']) > 20:
                    remaining = len(result['worker_counts']) - 20
                    print(f"  ... and {remaining} more workers")

            # Show sample jobs if verbose
            if args.verbose and result['jobs']:
                print(f"\nJob details (sample):")
                for job in result['jobs'][:50]:
                    hours = job.get('stuck_duration_hours')
                    duration_str = f"{hours:.1f}h" if hours else "N/A"
                    print(f"  Job #{job['id']}: {job['status']} for {duration_str}")
                    print(f"    URL: {job['episode_url']}")
                    print(f"    Worker: {job['worker_id']}")
                    print(f"    Assigned: {job['assigned_at']}")
                    print()

                if len(result['jobs']) > 50:
                    print(f"  ... and {len(result['jobs']) - 50} more jobs")

            if result['dry_run']:
                print(f"\n{'='*80}")
                print(f"This was a DRY RUN - no changes were made")
                print(f"Run with --execute to actually clean up these jobs")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"Successfully cleaned up {result['cleaned']:,} stuck jobs")
                print(f"These jobs are now back in PENDING status")
                print(f"{'='*80}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
