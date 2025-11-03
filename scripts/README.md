# Podserver Scripts

Administrative and utility scripts for managing the Podcast Pile system.

## invalidate_recent_jobs.py

Invalidate and mark for reprocessing all jobs processed after a specific date. This is useful when a bug in the worker processing code is discovered and completed jobs need to be reprocessed.

### Usage

**Dry run (default)** - Shows what would be invalidated without making changes:
```bash
python scripts/invalidate_recent_jobs.py --after "2025-10-29"
```

**Execute the invalidation:**
```bash
python scripts/invalidate_recent_jobs.py --after "2025-10-29" --execute
```

**With verbose output:**
```bash
python scripts/invalidate_recent_jobs.py --after "2025-10-29" --verbose --execute
```

**Specify a datetime with time component:**
```bash
python scripts/invalidate_recent_jobs.py --after "2025-10-29 15:30:00" --execute
```

### What it does

1. Queries the database for all jobs with status `COMPLETED` that were processed after the specified date
2. Resets these jobs to `PENDING` status
3. Clears all processing results (transcription, diarization, result_json)
4. Clears worker assignment information
5. Adds a note to the error_message field documenting the invalidation
6. Increments the retry_count to track reprocessing

### Options

- `--after DATE`: **(Required)** Invalidate jobs processed after this date. Accepts formats:
  - `YYYY-MM-DD`
  - `YYYY-MM-DD HH:MM:SS`
  - `YYYY-MM-DD HH:MM`
  - `YYYY/MM/DD`
  - `YYYY/MM/DD HH:MM:SS`

- `--execute`: Actually perform the invalidation. Without this flag, the script runs in dry-run mode.

- `--verbose` / `-v`: Show detailed information about each job.

- `--batch-size N`: Number of jobs to process per batch (default: 1000). Increase for faster processing on powerful servers, decrease to reduce memory usage.

### Performance

The script uses **batched processing** to efficiently handle large datasets:
- Processes jobs in configurable batch sizes (default: 1000)
- Commits each batch separately to avoid long-running transactions
- Shows progress reporting for long operations
- Memory-efficient: queries only necessary fields until update time
- In dry-run mode, scans up to 10,000 jobs for statistics to keep it fast

**Expected performance:**
- Dry run: ~5,000-10,000 jobs/sec (scanning only)
- Execution: ~500-2,000 jobs/sec (depending on database and batch size)
- For 100,000 jobs: ~1-3 minutes with default settings

### Safety Features

- **Dry-run by default**: Won't make changes unless you use `--execute`
- **Batched commits**: Each batch is committed separately, reducing risk
- **Detailed reporting**: Shows affected jobs grouped by podcast
- **Preserves history**: Documents invalidation in error_message field
- **Transaction safety**: Uses database transactions with rollback on error
- **Progress tracking**: Shows real-time progress during execution

---

## cleanup.py

Clean up stuck jobs that have been processing for too long. This script finds jobs in `PROCESSING` or `ASSIGNED` status that have exceeded their timeout and resets them to `PENDING` so they can be picked up again.

### Usage

**Dry run (default)** - Shows what would be cleaned up:
```bash
python scripts/cleanup.py
```

**Execute the cleanup:**
```bash
python scripts/cleanup.py --execute
```

**Custom timeout** (default is 2 hours):
```bash
python scripts/cleanup.py --timeout 4 --execute
```

**Clean up only specific status:**
```bash
# Only PROCESSING jobs
python scripts/cleanup.py --status PROCESSING --execute

# Only ASSIGNED jobs
python scripts/cleanup.py --status ASSIGNED --execute
```

**With verbose output:**
```bash
python scripts/cleanup.py --verbose
```

### What it does

1. Finds jobs in `PROCESSING` or `ASSIGNED` status that have been stuck for longer than the timeout
2. Checks both:
   - Jobs where `assigned_at` is older than the timeout
   - Jobs where `expires_at` has passed (for jobs with explicit expiration)
3. Resets these jobs to `PENDING` status
4. Clears worker assignment information
5. Documents the cleanup in `error_message` field for tracking
6. Increments `retry_count` to track retries

### When to use this

- **Worker crashes**: Workers that crashed without reporting failures
- **Network issues**: Workers that disconnected and couldn't complete jobs
- **Stuck jobs**: Jobs that got stuck for unknown reasons
- **Scheduled cleanup**: Run periodically (e.g., via cron) to automatically recover from failures

### Options

- `--timeout HOURS`: Reset jobs older than this many hours (default: 2.0)
- `--execute`: Actually perform the cleanup (without this, it's dry-run)
- `--status {PROCESSING,ASSIGNED,BOTH}`: Which status to check (default: BOTH)
- `--batch-size N`: Jobs to process per batch (default: 1000)
- `--verbose` / `-v`: Show detailed information about each job

### Performance

Uses the same batched processing approach as `invalidate_recent_jobs.py`:
- Memory efficient: queries only essential fields
- Batched commits: each batch committed separately
- Progress reporting during execution
- Fast dry-run: scans up to 10,000 jobs for statistics

**Expected performance:**
- Dry run: ~5,000-10,000 jobs/sec
- Execution: ~500-2,000 jobs/sec

### Running as a cron job

You can schedule this to run automatically:

```bash
# Run every 30 minutes to clean up stuck jobs
*/30 * * * * cd /path/to/podserver && python scripts/cleanup.py --execute >> /var/log/podserver-cleanup.log 2>&1
```

### Safety Features

- **Dry-run by default**: Won't make changes without `--execute`
- **Batched commits**: Reduces risk of long-running transactions
- **Detailed reporting**: Shows affected jobs grouped by status and worker
- **Preserves history**: Documents cleanup in error_message field
- **Transaction safety**: Rollback on error

---

## Import Scripts

Fast bulk import tools for loading large JSONL files into the Podcast Pile database.

## Quick Start

### For PostgreSQL (Recommended for 2M+ rows)

```bash
# Ultra-fast import using PostgreSQL COPY
python scripts/import_jobs_postgres.py jobs.jsonl
```

**Expected performance**: ~50,000-200,000 rows/sec
- 2M rows: ~10-40 seconds
- Uses PostgreSQL COPY command (100x faster than inserts)

### For SQLite or General Use

```bash
# Standard batch import
python scripts/import_jobs.py jobs.jsonl

# Custom batch size
python scripts/import_jobs.py jobs.jsonl --batch-size 5000

# Don't skip duplicates (will error on duplicates)
python scripts/import_jobs.py jobs.jsonl --no-skip-existing
```

**Expected performance**: ~1,000-5,000 rows/sec
- 2M rows: ~7-30 minutes
- Works with any database (SQLite, PostgreSQL, MySQL)

## JSONL Format

Your input file should have one JSON object per line:

```jsonl
{"duration_seconds": 3600, "episode_url": "https://example.com/ep1.mp3", "language": "en-US", "podcast_id": "123", "title": "Episode 1"}
{"duration_seconds": 1800, "episode_url": "https://example.com/ep2.mp3", "language": "zh-CN", "podcast_id": "123", "title": "Episode 2"}
```

**Required fields**:
- `episode_url` - URL to the episode audio file

**Optional fields**:
- `language` - Language code (e.g., "en-US", "zh-CN") - will be normalized to 2 letters
- `duration_seconds` - Not currently stored in DB (can be added if needed)
- `podcast_id` - Not currently stored in DB (can be added if needed)
- `title` - Not currently stored in DB (can be added if needed)

**Language normalization**:
- `"en-US"` → `"en"`
- `"zh-CN"` → `"zh"`
- `"es-MX"` → `"es"`

## Features

### Duplicate Detection

Both scripts automatically skip URLs that already exist in the database:
- Checks `episode_url` column for duplicates
- No errors thrown on duplicates
- Reports count of skipped rows

### Progress Reporting

**import_jobs.py**: Shows progress every 10,000 rows
```
Processed: 10,000 | Inserted: 9,850 | Skipped: 150 | Rate: 2,450 rows/sec
Processed: 20,000 | Inserted: 19,700 | Skipped: 300 | Rate: 2,500 rows/sec
```

**import_jobs_postgres.py**: Shows major milestones
```
Converting JSONL to CSV...
  Converted 100,000 rows...
  Converted 200,000 rows...
✓ Converted 2,000,000 rows in 5.2s
```

## Performance Comparison

For 2 million rows:

| Method | SQLite | PostgreSQL |
|--------|--------|------------|
| `import_jobs_postgres.py` | ❌ Not supported | ⚡ ~10-40s |
| `import_jobs.py` | 🐢 ~20-30min | 🐢 ~7-15min |

**Recommendation**: Use `import_jobs_postgres.py` for PostgreSQL with large datasets.

## Options

### import_jobs.py

```
--batch-size N          Records per batch (default: 1000)
--no-skip-existing      Don't skip existing URLs
--progress-interval N   Report every N records (default: 10000)
```

### import_jobs_postgres.py

No options - optimized for maximum speed by default.

## Error Handling

- **Invalid JSON**: Skipped with warning, continues processing
- **Missing episode_url**: Row skipped
- **Database errors**: Transaction rolled back, error reported
- **Ctrl+C**: Gracefully stops, commits completed batches

## Examples

### Import 2M jobs
```bash
python scripts/import_jobs_postgres.py data/all_episodes.jsonl
```

### Import with custom batch size
```bash
python scripts/import_jobs.py data/episodes.jsonl --batch-size 5000
```

### Import and allow duplicates (will error)
```bash
python scripts/import_jobs.py data/episodes.jsonl --no-skip-existing
```

## Troubleshooting

### "Error: This script requires PostgreSQL"
You're trying to use `import_jobs_postgres.py` with SQLite. Use `import_jobs.py` instead.

### Slow import speed
- **PostgreSQL**: Use `import_jobs_postgres.py` instead of `import_jobs.py`
- **SQLite**: Increase `--batch-size` to 5000 or 10000
- Check disk I/O - importing is I/O bound

### Out of memory
- Use `import_jobs.py` with smaller `--batch-size` (e.g., 500)
- Process file in chunks using `head`/`tail` commands

### Duplicate key errors
The default behavior is to skip duplicates. If you see these errors:
- Make sure you're not using `--no-skip-existing`
- Check if there are duplicates within the JSONL file itself
