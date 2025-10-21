from fastapi import Header, HTTPException
from typing import Optional
from ..config import config


async def verify_worker_auth(x_worker_password: Optional[str] = Header(None)):
    """Verify worker authentication if enabled."""
    if not config.WORKER_AUTH_ENABLED:
        return True

    if not x_worker_password:
        raise HTTPException(
            status_code=401,
            detail="Worker authentication required. Include X-Worker-Password header."
        )

    if x_worker_password != config.WORKER_PASSWORD:
        raise HTTPException(
            status_code=403,
            detail="Invalid worker password"
        )

    return True


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
