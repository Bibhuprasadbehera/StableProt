"""App-level controls for the ILS VAPT findings we can close without TLS/certs."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'self'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": CSP,
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

INFER_PATHS = {
    "/predict",
    "/api/predict",
    "/api/identify_loops",
    "/api/evaluate_design",
    "/api/random_loop_mutation",
}
SKIP_RATE_PATHS = {"/health"}


def allowed_hosts():
    raw = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1,192.168.4.156")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or ["localhost"]


def client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        if not request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
        if "server" in response.headers:
            del response.headers["server"]
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, infer_limit=10, infer_window=60, global_limit=60, global_window=60):
        super().__init__(app)
        self.infer_limit = infer_limit
        self.infer_window = infer_window
        self.global_limit = global_limit
        self.global_window = global_window
        self._hits = defaultdict(deque)

    def _allow(self, key: str, limit: int, window: float):
        now = time.monotonic()
        q = self._hits[key]
        cutoff = now - window
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(window - (now - q[0])) + 1)
            return False, retry
        q.append(now)
        return True, 0

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith("/static") or path in SKIP_RATE_PATHS:
            return await call_next(request)
        ip = client_ip(request)
        ok, retry = self._allow(f"{ip}:all", self.global_limit, self.global_window)
        if ok and path in INFER_PATHS:
            ok, retry = self._allow(f"{ip}:infer", self.infer_limit, self.infer_window)
        if not ok:
            return JSONResponse(
                {"error": "Too many requests"},
                status_code=429,
                headers={"Retry-After": str(retry)},
            )
        return await call_next(request)


def install(app):
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())
