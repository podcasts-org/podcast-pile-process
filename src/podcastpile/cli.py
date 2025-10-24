import click
import uvicorn
import httpx
import logging
import socket


@click.group()
def cli():
    """Podcast Pile CLI - Data processing infrastructure for podcasts."""
    pass


@cli.command()
@click.option('-p', '--port', default=8000, help='Port to run on')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def manager(port, host, reload):
    """Run the manager server."""
    click.echo(f"Starting Podcast Pile Manager on {host}:{port}...")
    uvicorn.run(
        "podcastpile.manager.server:app",
        host=host,
        port=port,
        reload=reload
    )


@cli.command()
@click.option('-m', '--manager', required=True, help='Manager URL (e.g., http://localhost:8000)')
@click.option('-i', '--worker-id', help='Worker ID (default: hostname)')
@click.option('-p', '--password', envvar='WORKER_PASSWORD', help='Worker password (can also use WORKER_PASSWORD env var)')
@click.option('-l', '--languages', default='en', help='Comma-separated language codes to process (default: en)')
@click.option('-c', '--config',
              type=click.Choice(['very_high_latency', 'high_latency', 'low_latency', 'ultra_low_latency']),
              default='high_latency',
              help='Diarization configuration (default: high_latency)')
@click.option('--model', help='Path to custom .nemo model file')
@click.option('--once', is_flag=True, help='Process one job and exit')
@click.option('--poll-interval', default=10, type=int, help='Seconds between polling for jobs (default: 10)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def worker(manager, worker_id, password, languages, config, model, once, poll_interval, verbose):
    """Run a worker to process jobs."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Auto-generate worker ID if not provided
    if not worker_id:
        worker_id = socket.gethostname()

    click.echo(f"Starting Podcast Pile Worker")
    click.echo(f"  Worker ID: {worker_id}")
    click.echo(f"  Manager: {manager}")
    click.echo(f"  Languages: {languages}")
    click.echo(f"  Config: {config}")
    click.echo(f"  Mode: {'Single job' if once else 'Continuous'}")
    click.echo()

    # Import worker (done here to avoid loading heavy dependencies if not needed)
    try:
        from podcastpile.worker import PodcastPileWorker
    except ImportError as e:
        click.echo(f"Error: Failed to import worker dependencies: {e}", err=True)
        click.echo("\nMake sure you have installed the required dependencies:", err=True)
        click.echo("  pip install nemo_toolkit[asr] librosa soundfile", err=True)
        raise click.Abort()

    # Create worker instance
    try:
        worker_instance = PodcastPileWorker(
            manager_url=manager,
            worker_id=worker_id,
            worker_password=password,
            config=config,
            model_path=model
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
@click.argument('url')
@click.option('-m', '--manager', default='http://localhost:8000', help='Manager URL')
@click.option('--language', help='Language code (e.g., en, es, fr)')
def create(url, manager, language):
    """Create a new job."""
    try:
        payload = {"episode_url": url}
        if language:
            payload["language"] = language

        response = httpx.post(
            f"{manager}/api/jobs",
            json=payload
        )
        response.raise_for_status()
        job = response.json()
        click.echo(f"Created job #{job['id']}")
        click.echo(f"  URL: {job['episode_url']}")
        click.echo(f"  Status: {job['status']}")
        if job.get('language'):
            click.echo(f"  Language: {job['language']}")
    except Exception as e:
        click.echo(f"Error creating job: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('-s', '--status', help='Filter by status')
@click.option('-l', '--limit', type=int, help='Limit number of results')
@click.option('-m', '--manager', default='http://localhost:8000', help='Manager URL')
def list(status, limit, manager):
    """List jobs."""
    try:
        params = {}
        if status:
            params['status'] = status
        if limit:
            params['limit'] = limit

        response = httpx.get(
            f"{manager}/api/jobs",
            params=params
        )
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
@click.option('-m', '--manager', default='http://localhost:8000', help='Manager URL')
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


if __name__ == '__main__':
    main()
