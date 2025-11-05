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
def version():
    """Print worker processor version."""
    from podcastpile.worker.processor import WORKER_VERSION
    click.echo(f"Worker Version: {WORKER_VERSION}")


@cli.command()
@click.argument("audio_url")
@click.option(
    "-l",
    "--language",
    default="en",
    help="Language code (default: en)",
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
    help="Batch size for ASR transcription (default: 4)",
)
@click.option("--gpu", type=int, help="GPU device ID to use (e.g., 0, 1, 2)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def test_worker(audio_url, language, config, model, batch_size, gpu, verbose):
    """Test the worker processor on a single audio file without writing results.

    This command downloads and processes an audio file to benchmark performance
    and verify the processing pipeline. No results are saved to the manager.

    Example:
        ppcli test-worker https://example.com/test.mp3
        ppcli test-worker https://example.com/test.mp3 --gpu 0 --verbose
    """
    import logging
    import tempfile
    import time
    import json
    from pathlib import Path

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    click.echo("=" * 60)
    click.echo("🧪 WORKER PROCESSOR TEST MODE")
    click.echo("=" * 60)

    # Import worker dependencies
    try:
        from podcastpile.worker.processor import AudioProcessor, WORKER_VERSION
        import requests
    except ImportError as e:
        click.echo(f"❌ Error: Failed to import dependencies: {e}", err=True)
        click.echo("\nInstall required dependencies:", err=True)
        click.echo("  pip install nemo_toolkit[asr] librosa soundfile torchaudio", err=True)
        raise click.Abort()

    click.echo(f"\n📋 Configuration:")
    click.echo(f"  Worker Version:  {WORKER_VERSION}")
    click.echo(f"  Audio URL:       {audio_url}")
    click.echo(f"  Language:        {language}")
    click.echo(f"  Diar Config:     {config}")
    click.echo(f"  Batch Size:      {batch_size}")
    click.echo(f"  GPU:             {gpu if gpu is not None else 'auto'}")
    click.echo()

    # Create processor
    click.echo("⏳ Initializing processor...")
    try:
        processor = AudioProcessor(
            config=config,
            model_path=model,
            gpu_id=gpu,
            languages=language,
            batch_size=batch_size,
        )
    except Exception as e:
        click.echo(f"❌ Failed to create processor: {e}", err=True)
        raise click.Abort()

    # Load models
    click.echo("⏳ Loading models (this may take a while)...")
    load_start = time.time()
    try:
        processor.load_models()
        load_time = time.time() - load_start
        click.echo(f"✅ Models loaded in {load_time:.1f}s\n")
    except Exception as e:
        click.echo(f"❌ Failed to load models: {e}", err=True)
        raise click.Abort()

    # Download audio
    click.echo(f"⏳ Downloading audio from {audio_url}...")
    download_start = time.time()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download file
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()

            # Determine file extension
            ext = ".mp3"
            if audio_url.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                ext = Path(audio_url).suffix

            audio_path = Path(temp_dir) / f"test_audio{ext}"

            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        click.echo(f"\r  Progress: {pct:.1f}%", nl=False)

            click.echo()  # New line after progress
            download_time = time.time() - download_start
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            click.echo(f"✅ Downloaded {size_mb:.1f}MB in {download_time:.1f}s\n")

            # Process audio
            click.echo("🚀 Processing audio...")
            click.echo("-" * 60)
            process_start = time.time()

            try:
                results = processor.diarize_audio(
                    audio_path=str(audio_path),
                    episode_url=audio_url,
                    language=language,
                )
                process_time = time.time() - process_start

                click.echo("-" * 60)
                click.echo(f"✅ Processing complete in {process_time:.1f}s\n")

                # Display results summary
                click.echo("=" * 60)
                click.echo("📊 PROCESSING RESULTS")
                click.echo("=" * 60)
                click.echo(f"\n⏱️  Performance:")
                click.echo(f"  Total Processing:    {process_time:.2f}s")
                click.echo(f"  Audio Duration:      {results['duration']:.2f}s ({results['duration']/60:.2f}min)")
                click.echo(f"  Real-time Factor:    {results['duration']/process_time:.2f}x")
                click.echo(f"  GPU Info:            {results['gpu_info'] or 'CPU'}")

                click.echo(f"\n🎙️  Diarization:")
                click.echo(f"  Total Segments:      {results['num_segments']}")
                click.echo(f"  Unique Speakers:     {results['num_speakers']}")
                click.echo(f"  Avg Segment Length:  {results['duration']/results['num_segments']:.2f}s")

                if results.get('episode_quality'):
                    eq = results['episode_quality']
                    click.echo(f"\n🎵 Audio Quality (NISQA):")
                    click.echo(f"  Mean MOS:            {eq['mean_mos']:.2f}")
                    click.echo(f"  Median MOS:          {eq['median_mos']:.2f}")
                    click.echo(f"  High Quality Segs:   {eq['high_quality_segments']}")
                    click.echo(f"  Medium Quality Segs: {eq['medium_quality_segments']}")
                    click.echo(f"  Low Quality Segs:    {eq['low_quality_segments']}")

                if results.get('clipping_stats'):
                    cs = results['clipping_stats']
                    click.echo(f"\n📉 Clipping Stats:")
                    click.echo(f"  Segments w/ Clip:    {cs['segments_with_clipping']}/{results['num_segments']}")
                    click.echo(f"  Max Clip Rate:       {cs['max_clip_rate']:.4f}")

                if results.get('loudness_stats'):
                    ls = results['loudness_stats']
                    click.echo(f"\n🔊 Loudness Stats:")
                    click.echo(f"  Mean RMS:            {ls['mean_rms_db']:.1f} dB")
                    click.echo(f"  Mean Peak:           {ls['mean_peak_db']:.1f} dB")
                    click.echo(f"  Dynamic Range:       {ls['mean_dynamic_range_db']:.1f} dB")

                # Sample transcriptions
                click.echo(f"\n📝 Sample Transcriptions (first 3 segments):")
                for i, seg in enumerate(results['segments'][:3]):
                    text = seg.get('transcription', '').strip()
                    if text:
                        click.echo(f"  [{seg['start']:.1f}s] Speaker {seg['speaker']}: {text[:80]}{'...' if len(text) > 80 else ''}")

                click.echo("\n" + "=" * 60)
                click.echo("✅ TEST COMPLETE")
                click.echo("=" * 60)

                # Offer to save results
                if click.confirm("\n💾 Save results to JSON file?", default=False):
                    output_file = f"test_results_{int(time.time())}.json"
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    click.echo(f"✅ Results saved to {output_file}")
                else:
                    click.echo("⚠️  Results not saved (test mode)")

            except Exception as e:
                click.echo(f"\n❌ Processing failed: {e}", err=True)
                logger.exception("Processing error")
                raise click.Abort()

    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Download failed: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        logger.exception("Test error")
        raise click.Abort()


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
