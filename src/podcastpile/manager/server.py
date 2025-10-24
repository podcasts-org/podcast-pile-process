from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import os

from ..models import get_db, Job, JobStatus, init_db
from ..config import config
from .auth import verify_worker_auth, verify_admin_auth, get_client_ip

app = FastAPI(title="Podcast Pile Manager", version="0.1.0")

# Get the templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Pydantic models for API
class JobCreate(BaseModel):
    episode_url: str
    podcast_id: Optional[str] = None  # External podcast identifier
    language: Optional[str] = None  # ISO 639-1 language code (e.g., 'en', 'es', 'fr')


class JobResponse(BaseModel):
    id: int
    episode_url: str
    podcast_id: Optional[str]
    status: JobStatus
    worker_id: Optional[str]
    language: Optional[str]
    created_at: datetime
    assigned_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class JobResult(BaseModel):
    transcription: Optional[str] = None
    diarization: Optional[str] = None
    result_json: Optional[str] = None
    processing_duration: Optional[float] = None  # Seconds to process
    worker_gpu: Optional[str] = None  # GPU device info
    processed_at: Optional[str] = None  # ISO format timestamp


class StatsResponse(BaseModel):
    total_jobs: int
    pending: int
    assigned: int
    processing: int
    completed: int
    failed: int
    expired: int


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    config.validate()
    init_db()


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    _: str = Depends(verify_admin_auth)
):
    """Render admin dashboard (optimized for large datasets)."""
    # Get statistics with single grouped query instead of 7 separate queries
    # Uses index: status
    status_counts = db.query(
        Job.status,
        func.count(Job.id).label('count')
    ).group_by(Job.status).all()

    stats_dict = {status.value: count for status, count in status_counts}

    stats = {
        "total": sum(stats_dict.values()),
        "pending": stats_dict.get(JobStatus.PENDING.value, 0),
        "assigned": stats_dict.get(JobStatus.ASSIGNED.value, 0),
        "processing": stats_dict.get(JobStatus.PROCESSING.value, 0),
        "completed": stats_dict.get(JobStatus.COMPLETED.value, 0),
        "failed": stats_dict.get(JobStatus.FAILED.value, 0),
        "expired": stats_dict.get(JobStatus.EXPIRED.value, 0),
    }

    # Get recent jobs (limit to 20, processing jobs first)
    recent_jobs = db.query(Job).order_by(
        (Job.status == JobStatus.PROCESSING).desc(),
        Job.created_at.desc()
    ).limit(20).all()

    # Get active workers count efficiently (uses index: ix_worker_status)
    active_worker_count = db.query(Job.worker_id).filter(
        Job.status.in_([JobStatus.ASSIGNED, JobStatus.PROCESSING]),
        Job.worker_id.isnot(None)
    ).distinct().count()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "recent_jobs": recent_jobs,
            "active_workers": active_worker_count,
        }
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """Get job statistics (optimized for large datasets)."""
    # Single grouped query instead of 7 separate queries
    status_counts = db.query(
        Job.status,
        func.count(Job.id).label('count')
    ).group_by(Job.status).all()

    stats_dict = {status.value: count for status, count in status_counts}

    return StatsResponse(
        total_jobs=sum(stats_dict.values()),
        pending=stats_dict.get(JobStatus.PENDING.value, 0),
        assigned=stats_dict.get(JobStatus.ASSIGNED.value, 0),
        processing=stats_dict.get(JobStatus.PROCESSING.value, 0),
        completed=stats_dict.get(JobStatus.COMPLETED.value, 0),
        failed=stats_dict.get(JobStatus.FAILED.value, 0),
        expired=stats_dict.get(JobStatus.EXPIRED.value, 0),
    )


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(job: JobCreate, db: Session = Depends(get_db)):
    """Create a new job."""
    db_job = Job(episode_url=job.episode_url, language=job.language)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.get("/api/jobs/paginated")
async def get_jobs_paginated_api(
    page: int = 1,
    per_page: int = 50,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    _: str = Depends(verify_admin_auth)
):
    """Get paginated jobs list (optimized for millions of jobs)."""
    # Validate pagination parameters
    per_page = min(per_page, 100)  # Max 100 per page
    offset = (page - 1) * per_page

    # Build query
    query = db.query(Job)

    if status_filter:
        try:
            status_enum = JobStatus(status_filter)
            query = query.filter(Job.status == status_enum)
        except ValueError:
            pass

    # Get total count (uses index)
    total = query.count()

    # Get page of jobs (uses index: created_at)
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(per_page).all()

    total_pages = (total + per_page - 1) // per_page

    return {
        "jobs": [
            {
                "id": job.id,
                "episode_url": job.episode_url,
                "status": job.status.value,
                "worker_id": job.worker_id,
                "worker_ip": job.worker_ip,
                "language": job.language,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "has_json": job.result_json is not None
            }
            for job in jobs
        ],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages
        }
    }


@app.get("/api/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List jobs with optional status filter."""
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()
    return jobs


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get a specific job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/request")
async def request_job(
    request: Request,
    worker_id: str,
    languages: Optional[str] = None,  # Comma-separated language codes (e.g., "en,es,fr")
    db: Session = Depends(get_db),
    _: bool = Depends(verify_worker_auth)
):
    """Worker requests a job to process.

    Args:
        worker_id: Unique identifier for the worker
        languages: Optional comma-separated list of language codes the worker can handle.
                  If provided, only jobs matching these languages (or jobs with no language set) will be returned.
                  Example: "en" or "en,es,fr"
    """
    # Expire old jobs first
    expired_jobs = db.query(Job).filter(
        Job.status.in_([JobStatus.ASSIGNED, JobStatus.PROCESSING]),
        Job.expires_at < datetime.utcnow()
    ).all()

    for job in expired_jobs:
        job.status = JobStatus.EXPIRED
        job.worker_id = None
    db.commit()

    # Build query for pending jobs
    query = db.query(Job).filter(Job.status == JobStatus.PENDING)

    # Apply language filter if worker specified languages
    if languages:
        language_list = [lang.strip() for lang in languages.split(',')]
        # Match jobs with specified languages OR jobs with no language set (NULL)
        # Uses index: ix_status_language
        query = query.filter(
            (Job.language.in_(language_list)) | (Job.language.is_(None))
        )

    # Find next pending job (ordered by creation time for fairness)
    job = query.order_by(Job.created_at.asc()).first()

    if not job:
        if languages:
            raise HTTPException(
                status_code=404,
                detail=f"No jobs available for languages: {languages}"
            )
        raise HTTPException(status_code=404, detail="No jobs available")

    # Get worker IP and assign job
    worker_ip = get_client_ip(request)
    job.assign_to_worker(worker_id, worker_ip)
    db.commit()
    db.refresh(job)

    return {
        "job_id": job.id,
        "episode_url": job.episode_url,
        "language": job.language,
        "expires_at": job.expires_at
    }


@app.post("/api/jobs/{job_id}/start")
async def start_job(
    job_id: int,
    worker_id: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_worker_auth)
):
    """Worker marks job as processing."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Job not assigned to this worker")

    if job.is_expired():
        job.status = JobStatus.EXPIRED
        db.commit()
        raise HTTPException(status_code=410, detail="Job expired")

    job.mark_processing()
    db.commit()
    return {"status": "processing"}


@app.post("/api/jobs/{job_id}/complete")
async def complete_job(
    job_id: int,
    worker_id: str,
    result: JobResult,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_worker_auth)
):
    """Worker submits completed job results."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Job not assigned to this worker")

    # Parse processed_at if provided
    processed_at = None
    if result.processed_at:
        from datetime import datetime
        try:
            processed_at = datetime.fromisoformat(result.processed_at.replace('Z', '+00:00'))
        except Exception:
            pass  # Ignore if parsing fails

    job.mark_completed(
        transcription=result.transcription,
        diarization=result.diarization,
        result_json=result.result_json,
        processing_duration=result.processing_duration,
        worker_gpu=result.worker_gpu,
        processed_at=processed_at
    )
    db.commit()
    return {"status": "completed"}


@app.post("/api/jobs/{job_id}/fail")
async def fail_job(
    job_id: int,
    worker_id: str,
    error_message: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_worker_auth)
):
    """Worker reports job failure."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Job not assigned to this worker")

    job.mark_failed(error_message)
    db.commit()
    return {"status": "failed"}


@app.get("/api/charts/data")
async def get_chart_data(
    db: Session = Depends(get_db),
    _: str = Depends(verify_admin_auth)
):
    """Get data for dashboard charts (optimized for large datasets)."""
    from datetime import timedelta

    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)

    # Throughput: jobs completed per hour (only query completed_at, not full rows)
    # Uses index: ix_status_completed
    completed_jobs = db.query(Job.completed_at).filter(
        Job.status == JobStatus.COMPLETED,
        Job.completed_at >= last_24h
    ).all()

    # Group by hour
    hourly_counts = {}
    for (completed_at,) in completed_jobs:
        hour_key = completed_at.replace(minute=0, second=0, microsecond=0)
        hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

    throughput_labels = []
    throughput_data = []
    for i in range(24):
        hour = now - timedelta(hours=23-i)
        hour_key = hour.replace(minute=0, second=0, microsecond=0)
        throughput_labels.append(hour_key.strftime("%H:%M"))
        throughput_data.append(hourly_counts.get(hour_key, 0))

    # Worker performance: jobs completed per worker (aggregated in DB)
    # Uses index: ix_worker_status
    worker_stats = db.query(
        Job.worker_id,
        func.count(Job.id).label('completed_count')
    ).filter(
        Job.status == JobStatus.COMPLETED,
        Job.worker_id.isnot(None)
    ).group_by(Job.worker_id).limit(50).all()  # Limit to top 50 workers

    worker_labels = [w[0] for w in worker_stats]
    worker_data = [w[1] for w in worker_stats]

    # Processing times: ONLY fetch last 20 jobs (ordered by ID descending)
    # Uses primary key index for fast retrieval
    recent_completed = db.query(
        Job.id,
        Job.assigned_at,
        Job.completed_at
    ).filter(
        Job.status == JobStatus.COMPLETED,
        Job.assigned_at.isnot(None),
        Job.completed_at.isnot(None)
    ).order_by(Job.id.desc()).limit(20).all()

    avg_times = []
    avg_time_labels = []

    for job_id, assigned_at, completed_at in reversed(recent_completed):
        processing_time = (completed_at - assigned_at).total_seconds() / 60
        avg_times.append(round(processing_time, 2))
        avg_time_labels.append(f"#{job_id}")

    # Overall average processing time: Use SQL AVG for efficiency
    # Calculate average using database-level aggregation (much faster)
    avg_result = db.query(
        func.avg(
            func.julianday(Job.completed_at) - func.julianday(Job.assigned_at)
        ).label('avg_days')
    ).filter(
        Job.status == JobStatus.COMPLETED,
        Job.assigned_at.isnot(None),
        Job.completed_at.isnot(None)
    ).first()

    # Convert from days to minutes
    total_avg_time = 0
    if avg_result and avg_result[0]:
        total_avg_time = round(avg_result[0] * 24 * 60, 2)

    # Status distribution: Use single query with grouping
    status_counts_query = db.query(
        Job.status,
        func.count(Job.id).label('count')
    ).group_by(Job.status).all()

    status_counts = {
        status.value: count
        for status, count in status_counts_query
        if count > 0
    }

    return {
        "throughput": {
            "labels": throughput_labels,
            "data": throughput_data
        },
        "workers": {
            "labels": worker_labels,
            "data": worker_data
        },
        "processing_times": {
            "labels": avg_time_labels,
            "data": avg_times
        },
        "avg_processing_time": total_avg_time,
        "status_distribution": status_counts
    }


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_list(
    request: Request,
    page: int = 1,
    status_filter: Optional[str] = None,
    _: str = Depends(verify_admin_auth)
):
    """Paginated jobs list page."""
    return templates.TemplateResponse(
        "jobs_list.html",
        {
            "request": request,
            "page": page,
            "status_filter": status_filter
        }
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(
    request: Request,
    job_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(verify_admin_auth)
):
    """Job detail page with JSON viewer."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return templates.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "job": job
        }
    )


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(
    job_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(verify_admin_auth)
):
    """Admin endpoint to retry a failed or expired job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Only allow retrying failed or expired jobs
    if job.status not in [JobStatus.FAILED, JobStatus.EXPIRED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status '{job.status.value}'. Only 'failed' or 'expired' jobs can be retried."
        )

    # Reset job to pending state
    job.status = JobStatus.PENDING
    job.worker_id = None
    job.worker_ip = None
    job.assigned_at = None
    job.completed_at = None
    job.expires_at = None
    job.error_message = None
    job.retry_count += 1

    # Clear previous results (optional - you might want to keep them)
    # Uncomment if you want to clear old results on retry:
    # job.transcription = None
    # job.diarization = None
    # job.result_json = None

    db.commit()
    db.refresh(job)

    return {
        "status": "success",
        "message": f"Job #{job_id} has been reset to pending status",
        "job": {
            "id": job.id,
            "status": job.status.value,
            "retry_count": job.retry_count
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}
