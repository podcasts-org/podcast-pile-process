# Podcast Pile

Data processing infra for Podcast Pile.

## Architecture

The manager server maintains a queue of episodes to process. Worker servers request jobs from the manager, which assigns them tasks. Workers then process their assigned jobs and return results (transcription, diarization, and metadata) to the manager, which stores them in the database.

Jobs automatically expire after 2 hours if not completed. The manager tracks worker IP addresses and worker IDs for each job.

Workers download episodes from their URLs, perform diarization using Nvidia NeMo, transcribe audio using Parakeet, and send the results back to the manager.

## Installation

### Manager Server

For running the manager server only (no worker):

```bash
pip install -e .
```

### Worker

For running a worker, you need additional ML dependencies:

```bash
pip install -e ".[worker]"
```

This will install:
- librosa (audio processing)
- soundfile (audio I/O)
- nemo_toolkit[asr] (NeMo ASR models for English)

**For Chinese language support**, also install:
```bash
pip install fireredasr huggingface-hub
```

The FireRedASR model will be automatically downloaded from HuggingFace on first use.

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
- `WORKER_PASSWORD`: Password for worker authentication (required if worker auth enabled)
- `ADMIN_AUTH_ENABLED`: Enable admin dashboard authentication (default: false)
- `ADMIN_USERNAME`: Username for admin login (default: admin)
- `ADMIN_PASSWORD`: Password for admin login (required if admin auth enabled)
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

### Worker

Start a worker to process jobs:

```bash
# Basic usage (processes English jobs only by default)
ppcli am http://localhost:8000

# With worker ID and password
ppcli worker -m http://localhost:8000 -i my-worker -p worker-password

# Process multiple languages
ppcli worker -m http://localhost:8000 -l en,es,fr

# Use custom diarization configuration
ppcli worker -m http://localhost:8000 -c low_latency

# Process one job and exit
ppcli worker -m http://localhost:8000 --once

# Verbose logging
ppcli worker -m http://localhost:8000 -v

# GPU Selection - Use specific GPU
ppcli worker -m http://localhost:8000 --gpu 0

# Multi-GPU - Spawn worker on each available GPU
ppcli worker -m http://localhost:8000 --all-gpus

# Multi-GPU - Use specific GPUs (e.g., 0, 1, and 3)
ppcli worker -m http://localhost:8000 --gpus 0,1,3
```

Worker Options:
- `-m, --manager`: Manager URL (required)
- `-i, --worker-id`: Worker ID (default: hostname-gpu{N})
- `-p, --password`: Worker password (can also use WORKER_PASSWORD env var)
- `-l, --languages`: Comma-separated language codes to process (default: en)
- `-c, --config`: Diarization configuration - `very_high_latency`, `high_latency` (default), `low_latency`, `ultra_low_latency`
- `--model`: Path to custom .nemo model file
- `--gpu`: GPU device ID to use (e.g., 0, 1, 2)
- `--all-gpus`: Spawn a worker process on each available GPU
- `--gpus`: Comma-separated list of GPU IDs to use (e.g., "0,1,3")
- `--once`: Process one job and exit
- `--poll-interval`: Seconds between polling for jobs (default: 10)
- `-v, --verbose`: Enable verbose logging

The worker will:
1. **Load models once at startup** - Models are loaded based on selected languages:
   - **English/other**: Parakeet TDT (NeMo ASR)
   - **Chinese (zh)**: FireRedASR-AED-L
   - If languages contain `zh` or `cn`, FireRedASR is loaded
   - If languages contain non-Chinese codes, Parakeet is loaded
2. Request jobs from the manager (filtered by language)
3. Download the audio file
4. Perform diarization (always uses NeMo SortFormer)
5. Perform transcription using the appropriate model based on job language
6. Compute SHA256 and MD5 hashes of the audio file
7. Upload results (JSON, transcription, diarization, GPU info, processing time) back to the manager
8. Repeat continuously (unless `--once` is used)

**Multi-GPU Support:**
- When using `--all-gpus` or `--gpus`, each GPU gets its own worker process
- Each worker has a unique ID: `hostname-gpu0`, `hostname-gpu1`, etc.
- Press Ctrl+C once to gracefully stop all workers
- All workers share the same configuration and can process jobs in parallel

## CLI Commands

### Create a job

```bash
# Create a job with an episode URL
ppcli create https://example.com/podcast.mp3

# Create a job with language tag
ppcli create https://example.com/spanish-podcast.mp3 --language es
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

## Authentication

### Admin Dashboard Authentication

To protect the admin dashboard and charts with a password:

1. Set `ADMIN_AUTH_ENABLED=true` in `.env`
2. Set `ADMIN_USERNAME=admin` in `.env` (or your preferred username)
3. Set `ADMIN_PASSWORD=your-secure-password` in `.env`

When enabled, accessing the dashboard at `http://localhost:8000` will prompt for HTTP Basic Authentication credentials.

### Worker Authentication

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

**Note:** Admin and worker authentication are independent and can be enabled separately or together.

## Development

Run in development mode with auto-reload:

```bash
ppcli manager --reload
```