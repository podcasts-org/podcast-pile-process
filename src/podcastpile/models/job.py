from sqlalchemy import Column, Integer, String, DateTime, Text, Enum as SQLEnum, Index
from datetime import datetime, timedelta
from enum import Enum
from .database import Base


class JobStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    episode_url = Column(String, nullable=False)
    ia_url = Column(String, nullable=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    worker_id = Column(String, nullable=True, index=True)
    worker_ip = Column(String, nullable=True)
    language = Column(String, nullable=True, index=True)  # Language code (e.g., 'en', 'es', 'fr')

    # Results
    transcription = Column(Text, nullable=True)
    diarization = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)  # JSON results from worker

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    assigned_at = Column(DateTime, nullable=True, index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)

    # Composite indexes for common queries
    __table_args__ = (
        Index('ix_status_created', 'status', 'created_at'),
        Index('ix_status_completed', 'status', 'completed_at'),
        Index('ix_status_expires', 'status', 'expires_at'),
        Index('ix_worker_status', 'worker_id', 'status'),
        Index('ix_status_language', 'status', 'language'),  # For language-filtered worker queries
    )

    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    def is_expired(self) -> bool:
        """Check if job has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def assign_to_worker(self, worker_id: str, worker_ip: str = None):
        """Assign job to a worker."""
        self.status = JobStatus.ASSIGNED
        self.worker_id = worker_id
        self.worker_ip = worker_ip
        self.assigned_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=2)

    def mark_processing(self):
        """Mark job as being processed."""
        self.status = JobStatus.PROCESSING

    def mark_completed(self, transcription: str = None, diarization: str = None, ia_url: str = None, result_json: str = None):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if transcription:
            self.transcription = transcription
        if diarization:
            self.diarization = diarization
        if ia_url:
            self.ia_url = ia_url
        if result_json:
            self.result_json = result_json

    def mark_failed(self, error_message: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
