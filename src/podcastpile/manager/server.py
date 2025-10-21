from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import os

from ..models import get_db, Job, JobStatus, init_db
from ..config import config
from .auth import verify_worker_auth, get_client_ip

app = FastAPI(title="Podcast Pile Manager", version="0.1.0")

# Get the templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Pydantic models for API
class JobCreate(BaseModel):
    episode_url: str


class JobResponse(BaseModel):
    id: int
    episode_url: str
    ia_url: Optional[str]
    status: JobStatus
    worker_id: Optional[str]
    created_at: datetime
    assigned_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class JobResult(BaseModel):
    transcription: Optional[str] = None
    diarization: Optional[str] = None
    ia_url: Optional[str] = None


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
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Render admin dashboard."""
    # Get statistics
    stats = {
        "total": db.query(Job).count(),
        "pending": db.query(Job).filter(Job.status == JobStatus.PENDING).count(),
        "assigned": db.query(Job).filter(Job.status == JobStatus.ASSIGNED).count(),
        "processing": db.query(Job).filter(Job.status == JobStatus.PROCESSING).count(),
        "completed": db.query(Job).filter(Job.status == JobStatus.COMPLETED).count(),
        "failed": db.query(Job).filter(Job.status == JobStatus.FAILED).count(),
        "expired": db.query(Job).filter(Job.status == JobStatus.EXPIRED).count(),
    }

    # Get recent jobs
    recent_jobs = db.query(Job).order_by(Job.created_at.desc()).limit(20).all()

    # Get active workers
    active_workers = db.query(Job.worker_id).filter(
        Job.status.in_([JobStatus.ASSIGNED, JobStatus.PROCESSING])
    ).distinct().all()
    active_worker_count = len([w for w in active_workers if w[0] is not None])

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
    """Get job statistics."""
    return StatsResponse(
        total_jobs=db.query(Job).count(),
        pending=db.query(Job).filter(Job.status == JobStatus.PENDING).count(),
        assigned=db.query(Job).filter(Job.status == JobStatus.ASSIGNED).count(),
        processing=db.query(Job).filter(Job.status == JobStatus.PROCESSING).count(),
        completed=db.query(Job).filter(Job.status == JobStatus.COMPLETED).count(),
        failed=db.query(Job).filter(Job.status == JobStatus.FAILED).count(),
        expired=db.query(Job).filter(Job.status == JobStatus.EXPIRED).count(),
    )


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(job: JobCreate, db: Session = Depends(get_db)):
    """Create a new job."""
    db_job = Job(episode_url=job.episode_url)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


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
    db: Session = Depends(get_db),
    _: bool = Depends(verify_worker_auth)
):
    """Worker requests a job to process."""
    # Expire old jobs first
    expired_jobs = db.query(Job).filter(
        Job.status.in_([JobStatus.ASSIGNED, JobStatus.PROCESSING]),
        Job.expires_at < datetime.utcnow()
    ).all()

    for job in expired_jobs:
        job.status = JobStatus.EXPIRED
        job.worker_id = None
    db.commit()

    # Find next pending job
    job = db.query(Job).filter(Job.status == JobStatus.PENDING).first()

    if not job:
        raise HTTPException(status_code=404, detail="No jobs available")

    # Get worker IP and assign job
    worker_ip = get_client_ip(request)
    job.assign_to_worker(worker_id, worker_ip)
    db.commit()
    db.refresh(job)

    return {
        "job_id": job.id,
        "episode_url": job.episode_url,
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

    job.mark_completed(
        transcription=result.transcription,
        diarization=result.diarization,
        ia_url=result.ia_url
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}
