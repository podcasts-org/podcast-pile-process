import logging
import socket

import click
import httpx
import uvicorn
from dotenv import load_dotenv


@click.group()
def cli():
    """Podcast Pile CLI - Data processing infrastructure for podcasts."""
    pass


@cli.command()
@click.option("-p", "--port", default=8000, help="Port to run on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("-w", "--workers", default=4, type=int, help="Number of worker processes (default: 4, ignored with --reload)")
def manager(port, host, reload, workers):
    """Run the manager server."""
    # Load environment variables from .env file
    load_dotenv()

    if reload:
        click.echo(f"Starting Podcast Pile Manager on {host}:{port} (reload mode, single worker)...")
        uvicorn.run("podcastpile.manager.server:app", host=host, port=port, reload=reload)
    else:
        click.echo(f"Starting Podcast Pile Manager on {host}:{port} with {workers} workers...")
        uvicorn.run("podcastpile.manager.server:app", host=host, port=port, workers=workers)


@cli.command()
@click.option(
    "-m", "--manager", required=True, help="Manager URL (e.g., http://localhost:8000)"
)
@click.option("-i", "--worker-id", help="Worker ID (default: hostname-gpu{N})")
@click.option(
    "-p",
    "--password",
    envvar="WORKER_PASSWORD",
    help="Worker password (can also use WORKER_PASSWORD env var)",
)
@click.option(
    "-l",
    "--languages",
    default="en",
    help="Comma-separated language codes to process (default: en)",
)
@click.option(
    "-c",
    "--config",
    type=click.Choice(
        ["very_high_latency", "high_latency", "low_latency", "ultra_low_latency"]
    ),
    default="high_latency",
    help="Diarization configuration (default: high_latency)",
)
@click.option("--model", help="Path to custom .nemo model file")
@click.option(
    "--batch-size",
    default=4,
    type=int,
    help="Batch size for FireRedASR transcription (1, 2, 4, 8, 16, etc.) Default: 4",
)
@click.option("--once", is_flag=True, help="Process one job and exit")
@click.option(
    "--poll-interval",
    default=10,
    type=int,
    help="Seconds between polling for jobs (default: 10)",
)
@click.option("--gpu", type=int, help="GPU device ID to use (e.g., 0, 1, 2)")
@click.option("--all-gpus", is_flag=True, help="Spawn a worker on each available GPU")
@click.option("--gpus", help='Comma-separated list of GPU IDs to use (e.g., "0,1,3")')
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--s3-endpoint",
    envvar="S3_ENDPOINT_URL",
    default="https://s3.mrfake.name",
    help="S3 endpoint URL (default: https://s3.mrfake.name, can also use S3_ENDPOINT_URL env var)",
)
@click.option(
    "--s3-access-key",
    envvar="S3_ACCESS_KEY_ID",
    help="S3 access key ID (can also use S3_ACCESS_KEY_ID env var)",
)
@click.option(
    "--s3-secret-key",
    envvar="S3_SECRET_ACCESS_KEY",
    help="S3 secret access key (can also use S3_SECRET_ACCESS_KEY env var)",
)
@click.option(
    "--s3-bucket",
    envvar="S3_BUCKET",
    default="ppworker",
    help="S3 bucket name for audio uploads (default: ppworker, can also use S3_BUCKET env var)",
)
@click.option(
    "--s3-region",
    envvar="S3_REGION",
    default="us-east-1",
    help="S3 region (default: us-east-1, can also use S3_REGION env var)",
)
def worker(
    manager,
    worker_id,
    password,
    languages,
    config,
    model,
    batch_size,
    once,
    poll_interval,
    gpu,
    all_gpus,
    gpus,
    verbose,
    s3_endpoint,
    s3_access_key,
    s3_secret_key,
    s3_bucket,
    s3_region,
):
    """Run a worker to process jobs."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Import worker (done here to avoid loading heavy dependencies if not needed)
    try:
        from podcastpile.worker import PodcastPileWorker, get_available_gpus
    except ImportError as e:
        click.echo(f"Error: Failed to import worker dependencies: {e}", err=True)
        click.echo(
            "\nMake sure you have installed the required dependencies:", err=True
        )
        click.echo("  pip install nemo_toolkit[asr] librosa soundfile", err=True)
        raise click.Abort()

    # Determine which GPUs to use
    gpu_list = []
    if all_gpus:
        gpu_list = get_available_gpus()
        if not gpu_list:
            click.echo("No GPUs available!", err=True)
            raise click.Abort()
        click.echo(f"Using all available GPUs: {gpu_list}")
    elif gpus:
        gpu_list = [int(g.strip()) for g in gpus.split(",")]
        click.echo(f"Using specified GPUs: {gpu_list}")
    elif gpu is not None:
        gpu_list = [gpu]
        click.echo(f"Using GPU {gpu}")
    else:
        # Auto-detect available GPUs and spawn worker for each
        gpu_list = get_available_gpus()
        if not gpu_list:
            click.echo("No GPUs detected, running in CPU mode", err=True)
            gpu_list = [None]
        else:
            click.echo(f"Auto-detected GPUs: {gpu_list}")

    # Multi-GPU mode - spawn multiple workers
    if len(gpu_list) > 1:
        import multiprocessing
        import signal
        import sys

        click.echo(f"\nSpawning {len(gpu_list)} workers...")

        processes = []
        base_worker_id = worker_id or socket.gethostname()

        # Flag to track shutdown request (set by signal handler)
        shutdown_requested = False

        def signal_handler(sig, frame):
            # IMPORTANT: Signal handlers must be minimal to avoid:
            # 1. Reentrant I/O errors (no print/click.echo/logging)
            # 2. "can only join child process" errors (no process.join())
            # Just set a flag and let the main loop handle cleanup
            nonlocal shutdown_requested
            shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        for gpu_id in gpu_list:
            wid = f"{base_worker_id}-gpu{gpu_id}"
            p = multiprocessing.Process(
                target=_run_single_worker,
                args=(
                    manager,
                    wid,
                    password,
                    languages,
                    config,
                    model,
                    batch_size,
                    once,
                    poll_interval,
                    gpu_id,
                    verbose,
                    s3_endpoint,
                    s3_access_key,
                    s3_secret_key,
                    s3_bucket,
                    s3_region,
                ),
            )
            p.start()
            processes.append(p)
            click.echo(f"  ✓ Started worker {wid} (PID: {p.pid})")

        click.echo(
            f"\nAll {len(gpu_list)} workers running. Press Ctrl+C to stop all workers."
        )

        # Wait for processes or shutdown signal
        try:
            while not shutdown_requested:
                # Check if any process has died
                alive = [p for p in processes if p.is_alive()]
                if len(alive) < len(processes):
                    click.echo(f"\nWarning: {len(processes) - len(alive)} worker(s) died unexpectedly")
                    break

                # Sleep briefly to avoid busy waiting
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            # Shouldn't happen (signal handler should catch it), but just in case
            shutdown_requested = True

        # Graceful shutdown: terminate workers and wait briefly
        if shutdown_requested:
            click.echo("\n\nShutdown requested. Terminating workers...")
            click.echo("Note: Jobs in progress will remain in 'processing' state for manager cleanup")

            # Send terminate signal to all workers
            for p in processes:
                if p.is_alive():
                    p.terminate()

            # Wait a bit for graceful termination
            click.echo("Waiting for workers to stop...")
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    click.echo(f"Warning: Worker {p.pid} did not stop gracefully, killing...")
                    p.kill()

            click.echo("All workers stopped")
        else:
            # Workers died unexpectedly, just inform the user
            click.echo("One or more workers stopped unexpectedly. Check logs for details.")

    else:
        # Single worker mode
        gpu_id = gpu_list[0]
        if not worker_id:
            worker_id = (
                f"{socket.gethostname()}-gpu{gpu_id}"
                if gpu_id is not None
                else socket.gethostname()
            )

        _run_single_worker(
            manager,
            worker_id,
            password,
            languages,
            config,
            model,
            batch_size,
            once,
            poll_interval,
            gpu_id,
            verbose,
            s3_endpoint,
            s3_access_key,
            s3_secret_key,
            s3_bucket,
            s3_region,
        )


def _run_single_worker(
    manager,
    worker_id,
    password,
    languages,
    config,
    model,
    batch_size,
    once,
    poll_interval,
    gpu_id,
    verbose,
    s3_endpoint,
    s3_access_key,
    s3_secret_key,
    s3_bucket,
    s3_region,
):
    """Run a single worker instance"""
    import logging

    import click

    # Re-setup logging for subprocess
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s - [{worker_id}] - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    from podcastpile.worker import PodcastPileWorker

    click.echo(f"Starting Podcast Pile Worker: {worker_id}")
    click.echo(f"  Manager: {manager}")
    click.echo(f"  Languages: {languages}")
    click.echo(f"  Config: {config}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  GPU: {gpu_id if gpu_id is not None else 'auto'}")
    click.echo(f"  Mode: {'Single job' if once else 'Continuous'}")
    click.echo()

    # Create S3 config if credentials are provided
    s3_config = None
    if s3_access_key and s3_secret_key:
        s3_config = {
            "endpoint_url": s3_endpoint,
            "access_key_id": s3_access_key,
            "secret_access_key": s3_secret_key,
            "bucket": s3_bucket,
            "region": s3_region,
        }
        click.echo(f"  S3 uploads: Enabled (bucket: {s3_bucket})")
    else:
        click.echo("  S3 uploads: Disabled (no credentials provided)")

    # Create worker instance
    try:
        worker_instance = PodcastPileWorker(
            manager_url=manager,
            worker_id=worker_id,
            worker_password=password,
            config=config,
            model_path=model,
            gpu_id=gpu_id,
            languages=languages,
            batch_size=batch_size,
            s3_config=s3_config,
        )
    except Exception as e:
        click.echo(f"Error creating worker: {e}", err=True)
        raise click.Abort()

    # Load models
    click.echo("Loading models (this may take a while)...")
    try:
        worker_instance.load_models()
        click.echo("✓ Models loaded successfully")
        click.echo()
    except Exception as e:
        click.echo(f"Error loading models: {e}", err=True)
        raise click.Abort()

    # Run worker
    try:
        if once:
            # Process single job
            success = worker_instance.run_once(languages=languages)
            if success:
                click.echo("✓ Job processed successfully")
            else:
                click.echo("No jobs available")
        else:
            # Continuous mode
            worker_instance.run_loop(languages=languages, poll_interval=poll_interval)
    except KeyboardInterrupt:
        click.echo("\n\nWorker stopped by user")
    except Exception as e:
        click.echo(f"Error running worker: {e}", err=True)
        logger.exception("Worker error")
        raise click.Abort()


@cli.command()
@click.argument("url")
@click.option("-m", "--manager", default="https://podcastpile.mrfake.name", help="Manager URL")
@click.option("--language", help="Language code (e.g., en, es, fr)")
def create(url, manager, language):
    """Create a new job."""
    try:
        payload = {"episode_url": url}
        if language:
            payload["language"] = language

        response = httpx.post(f"{manager}/api/jobs", json=payload)
        response.raise_for_status()
        job = response.json()
        click.echo(f"Created job #{job['id']}")
        click.echo(f"  URL: {job['episode_url']}")
        click.echo(f"  Status: {job['status']}")
        if job.get("language"):
            click.echo(f"  Language: {job['language']}")
    except Exception as e:
        click.echo(f"Error creating job: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("-s", "--status", help="Filter by status")
@click.option("-l", "--limit", type=int, help="Limit number of results")
@click.option("-m", "--manager", default="http://localhost:8000", help="Manager URL")
def list(status, limit, manager):
    """List jobs."""
    try:
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit

        response = httpx.get(f"{manager}/api/jobs", params=params)
        response.raise_for_status()
        jobs = response.json()

        if not jobs:
            click.echo("No jobs found.")
            return

        click.echo(f"\nFound {len(jobs)} job(s):\n")
        for job in jobs:
            click.echo(f"Job #{job['id']}")
            click.echo(f"  URL: {job['episode_url']}")
            click.echo(f"  Status: {job['status']}")
            click.echo(f"  Worker: {job['worker_id'] or 'None'}")
            click.echo(f"  Created: {job['created_at']}")
            click.echo()
    except Exception as e:
        click.echo(f"Error listing jobs: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("-m", "--manager", default="http://localhost:8000", help="Manager URL")
def stats(manager):
    """Show statistics."""
    try:
        response = httpx.get(f"{manager}/api/stats")
        response.raise_for_status()
        stats = response.json()

        click.echo("\nPodcast Pile Statistics:")
        click.echo("=" * 40)
        click.echo(f"Total Jobs:    {stats['total_jobs']}")
        click.echo(f"Pending:       {stats['pending']}")
        click.echo(f"Assigned:      {stats['assigned']}")
        click.echo(f"Processing:    {stats['processing']}")
        click.echo(f"Completed:     {stats['completed']}")
        click.echo(f"Failed:        {stats['failed']}")
        click.echo(f"Expired:       {stats['expired']}")
        click.echo()
    except Exception as e:
        click.echo(f"Error getting stats: {e}", err=True)
        raise click.Abort()


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
