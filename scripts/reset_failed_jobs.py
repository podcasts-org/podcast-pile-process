#!/usr/bin/env python3
"""
Reset failed jobs to pending status for reprocessing.

This script marks all failed jobs as pending so they can be retried by workers.
Run this on the manager server.

Usage:
    python scripts/reset_failed_jobs.py [options]

Examples:
    # Reset all failed jobs
    python scripts/reset_failed_jobs.py

    # Dry run (preview what would be reset)
    python scripts/reset_failed_jobs.py --dry-run

    # Reset only jobs that failed with a specific error
    python scripts/reset_failed_jobs.py --error-contains "NISQA"

    # Reset only recent failures (last 24 hours)
    python scripts/reset_failed_jobs.py --hours 24
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from podcastpile.models import Job, JobStatus, SessionLocal


def reset_failed_jobs(
    dry_run: bool = False,
    error_contains: str = None,
    hours: int = None,
    limit: int = None,
):
    """
    Reset failed jobs to pending status.

    Args:
        dry_run: If True, only show what would be reset without making changes
        error_contains: Only reset jobs whose error message contains this text
        hours: Only reset jobs that failed in the last N hours
        limit: Maximum number of jobs to reset
    """
    # Use the existing database session
    session = SessionLocal()

    try:
        # Build query for failed jobs
        query = session.query(Job).filter(Job.status == JobStatus.FAILED)

        # Apply filters
        if error_contains:
            query = query.filter(Job.error_message.contains(error_contains))

        if hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = query.filter(Job.completed_at >= cutoff_time)

        # Order by ID and apply limit
        query = query.order_by(Job.id)
        if limit:
            query = query.limit(limit)

        # Get jobs to reset
        jobs = query.all()

        if not jobs:
            print("No failed jobs found matching the criteria.")
            return 0

        # Display summary
        print(f"\nFound {len(jobs)} failed job(s) to reset:")
        print("-" * 80)
        for job in jobs[:10]:  # Show first 10
            error_preview = job.error_message[:60] + "..." if job.error_message and len(job.error_message) > 60 else job.error_message or "No error message"
            print(f"  Job #{job.id}: {error_preview}")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        print("-" * 80)

        if dry_run:
            print("\n[DRY RUN] No changes made. Run without --dry-run to actually reset jobs.")
            return 0

        # Confirm with user
        print(f"\nThis will reset {len(jobs)} job(s) to PENDING status.")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

        # Reset jobs
        reset_count = 0
        for job in jobs:
            job.status = JobStatus.PENDING
            job.worker_id = None
            job.worker_ip = None
            job.assigned_at = None
            job.completed_at = None
            job.processing_duration = None
            job.error_message = None
            # Keep retry_count as is - it will track how many times it's been retried
            reset_count += 1

        session.commit()
        print(f"\n✓ Successfully reset {reset_count} job(s) to PENDING status.")
        return reset_count

    except Exception as e:
        session.rollback()
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return -1

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Reset failed jobs to pending status for reprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reset all failed jobs
  %(prog)s

  # Dry run to preview changes
  %(prog)s --dry-run

  # Reset only NISQA-related failures
  %(prog)s --error-contains "NISQA"

  # Reset only jobs that failed in the last 24 hours
  %(prog)s --hours 24

  # Reset maximum 10 jobs
  %(prog)s --limit 10

  # Combine filters
  %(prog)s --error-contains "403" --hours 48 --dry-run
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually resetting jobs'
    )
    parser.add_argument(
        '--error-contains',
        help='Only reset jobs whose error message contains this text'
    )
    parser.add_argument(
        '--hours',
        type=int,
        help='Only reset jobs that failed in the last N hours'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of jobs to reset'
    )

    args = parser.parse_args()

    # Run the reset
    result = reset_failed_jobs(
        dry_run=args.dry_run,
        error_contains=args.error_contains,
        hours=args.hours,
        limit=args.limit,
    )

    # Exit with appropriate code
    sys.exit(0 if result >= 0 else 1)


if __name__ == "__main__":
    main()
