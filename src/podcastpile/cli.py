import click
import uvicorn
import httpx


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
@click.option('-i', '--worker-id', help='Worker ID (default: auto-generated)')
def worker(manager, worker_id):
    """Run a worker (placeholder for now)."""
    click.echo("Worker functionality coming soon!")
    click.echo(f"Would connect to manager at: {manager}")
    click.echo("\nWorker implementation will include:")
    click.echo("  - Download episodes from URLs")
    click.echo("  - Perform diarization using Nvidia NeMo")
    click.echo("  - Transcribe audio using Parakeet")
    click.echo("  - Submit results back to manager")


@cli.command()
@click.argument('url')
@click.option('-m', '--manager', default='http://localhost:8000', help='Manager URL')
def create(url, manager):
    """Create a new job."""
    try:
        response = httpx.post(
            f"{manager}/api/jobs",
            json={"episode_url": url}
        )
        response.raise_for_status()
        job = response.json()
        click.echo(f"Created job #{job['id']}")
        click.echo(f"  URL: {job['episode_url']}")
        click.echo(f"  Status: {job['status']}")
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
