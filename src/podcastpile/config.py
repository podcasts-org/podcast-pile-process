import os
from typing import Optional


class Config:
    """Application configuration."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./podcastpile.db")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Worker Authentication
    WORKER_AUTH_ENABLED: bool = (
        os.getenv("WORKER_AUTH_ENABLED", "false").lower() == "true"
    )
    WORKER_PASSWORD: Optional[str] = os.getenv("WORKER_PASSWORD")

    # Admin Authentication
    ADMIN_AUTH_ENABLED: bool = (
        os.getenv("ADMIN_AUTH_ENABLED", "false").lower() == "true"
    )
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: Optional[str] = os.getenv("ADMIN_PASSWORD")

    # Job settings
    JOB_TIMEOUT_HOURS: int = int(os.getenv("JOB_TIMEOUT_HOURS", "2"))

    @classmethod
    def validate(cls):
        """Validate configuration."""
        if cls.WORKER_AUTH_ENABLED and not cls.WORKER_PASSWORD:
            raise ValueError(
                "WORKER_PASSWORD must be set when WORKER_AUTH_ENABLED is true"
            )
        if cls.ADMIN_AUTH_ENABLED and not cls.ADMIN_PASSWORD:
            raise ValueError(
                "ADMIN_PASSWORD must be set when ADMIN_AUTH_ENABLED is true"
            )


config = Config()
