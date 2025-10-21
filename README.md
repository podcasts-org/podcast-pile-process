# Podcast Pile

Data processing infra for Podcast Pile.

## Architecture

The manager server maintains a queue of episodes to process. Worker servers request jobs from the manager, which assigns them tasks. Workers then process their assigned jobs and return results (transcription, diarization, Internet Archive URL, etc.) to the manager, which stores them in the database.

Jobs automatically expire after 2 hours if not completed. The manager tracks worker IP addresses and worker IDs for each job.

Workers download episodes from their remote URLs, perform diarization using Nvidia NeMo, transcribe audio using Parakeet, and send the results back to the manager.

In the background, the manager submits URLs to the Internet Archive's Save Page Now API for archival. The manager only serves URLs from the Internet Archive rather than direct episode URLs (to avoid timestamp issues with dynamic ads).

## Installation

```bash
pip install -e .
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

Configuration options:
- `DATABASE_URL`: Database connection string (default: SQLite)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `WORKER_AUTH_ENABLED`: Enable worker authentication (default: false)
- `WORKER_PASSWORD`: Password for worker authentication (required if auth enabled)
- `JOB_TIMEOUT_HOURS`: Job timeout in hours (default: 2)

## Running

### Manager Server

Start the manager server:

```bash
ppcli manager -p 8000
```

Options:
- `-p, --port`: Port to run on (default: 8000)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--reload`: Enable auto-reload for development

The admin dashboard will be available at `http://localhost:8000`

### Worker (Coming Soon)

Worker functionality is planned but not yet implemented. When ready, workers will:
- Download episodes from URLs
- Perform diarization using Nvidia NeMo
- Transcribe audio using Parakeet
- Submit results back to manager

```bash
ppcli worker -m http://<manager-host>:8000
```

## CLI Commands

### Create a job

```bash
ppcli create https://example.com/podcast.mp3
```

### List jobs

```bash
# List all jobs
ppcli list

# Filter by status
ppcli list --status pending

# Limit results
ppcli list --limit 10
```

### View statistics

```bash
ppcli stats
```

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

- `GET /` - Admin dashboard
- `GET /api/stats` - Get job statistics
- `POST /api/jobs` - Create a new job
- `GET /api/jobs` - List jobs
- `POST /api/jobs/request` - Worker requests a job (requires auth if enabled)
- `POST /api/jobs/{job_id}/start` - Mark job as processing (requires auth if enabled)
- `POST /api/jobs/{job_id}/complete` - Submit job results (requires auth if enabled)
- `POST /api/jobs/{job_id}/fail` - Report job failure (requires auth if enabled)

## Worker Authentication

To enable worker authentication:

1. Set `WORKER_AUTH_ENABLED=true` in `.env`
2. Set `WORKER_PASSWORD=your-secure-password` in `.env`
3. Workers must include `X-Worker-Password` header in all authenticated requests

Example:

```bash
curl -X POST http://localhost:8000/api/jobs/request \
  -H "X-Worker-Password: your-secure-password" \
  -H "Content-Type: application/json" \
  -d '{"worker_id": "worker-1"}'
```

## Development

Run in development mode with auto-reload:

```bash
ppcli manager --reload
```