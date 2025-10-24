# Import Scripts

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
