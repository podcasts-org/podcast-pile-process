import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..config import config

security = HTTPBasic()


async def verify_worker_auth(x_worker_password: Optional[str] = Header(None)):
    """Verify worker authentication if enabled."""
    if not config.WORKER_AUTH_ENABLED:
        return True

    if not x_worker_password:
        raise HTTPException(
            status_code=401,
            detail="Worker authentication required. Include X-Worker-Password header.",
        )

    if x_worker_password != config.WORKER_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid worker password")

    return True


async def verify_admin_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(
        lambda: HTTPBasic(auto_error=False)
    )
):
    """Verify admin authentication if enabled."""
    if not config.ADMIN_AUTH_ENABLED:
        return True

    # If auth is enabled, credentials are required
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Ensure password is configured
    if not config.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Admin password not configured",
        )

    # Use constant-time comparison to prevent timing attacks
    correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"), config.ADMIN_USERNAME.encode("utf8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"), config.ADMIN_PASSWORD.encode("utf8")
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def get_client_ip(request) -> str:
    """Extract client IP address from request."""
    # Try X-Forwarded-For header first (for proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Try X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"
