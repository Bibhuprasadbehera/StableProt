"""App-level controls for the ILS VAPT findings we can close without TLS/certs."""
from __future__ import annotations

import os
import secrets
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
SESSION_COOKIE = "sp_session"


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


def session_id(request):
    sid = request.cookies.get(SESSION_COOKIE, "")
    if len(sid) == 32 and all(c in "0123456789abcdef" for c in sid):
        return sid, False
    return secrets.token_hex(16), True


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
    """10 inference requests per browser session per minute.

    Concurrent users each get their own session cookie, so ten people
    on the same lab IP do not share one bucket. A higher per-IP cap
    still stops cookie-reset floods (the VAPT 10k-request case).
    """

    def __init__(
        self,
        app,
        session_infer_limit=10,
        session_infer_window=60,
        session_all_limit=60,
        ip_infer_limit=150,
        ip_all_limit=300,
    ):
        super().__init__(app)
        self.session_infer_limit = session_infer_limit
        self.session_infer_window = session_infer_window
        self.session_all_limit = session_all_limit
        self.ip_infer_limit = ip_infer_limit
        self.ip_all_limit = ip_all_limit
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

    def _attach_cookie(self, response, sid: str, is_new: bool):
        if is_new:
            response.set_cookie(
                SESSION_COOKIE,
                sid,
                max_age=86400,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return response

    async def dispatch(self, request, call_next):
        path = request.url.path
        sid, is_new = session_id(request)
        if path.startswith("/static") or path in SKIP_RATE_PATHS:
            response = await call_next(request)
            return self._attach_cookie(response, sid, is_new)

        ip = client_ip(request)
        is_infer = path in INFER_PATHS
        checks = [
            (f"{ip}:all", self.ip_all_limit, 60),
            (f"{sid}:all", self.session_all_limit, 60),
        ]
        if is_infer:
            checks.extend([
                (f"{ip}:infer", self.ip_infer_limit, 60),
                (f"{sid}:infer", self.session_infer_limit, self.session_infer_window),
            ])

        retry = 1
        for key, limit, window in checks:
            ok, retry = self._allow(key, limit, window)
            if not ok:
                response = JSONResponse(
                    {"error": "Too many requests"},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
                return self._attach_cookie(response, sid, is_new)

        response = await call_next(request)
        return self._attach_cookie(response, sid, is_new)


def install(app):
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())
