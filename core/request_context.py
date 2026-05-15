"""
Request-scoped context for log correlation + latency timing.

Provides:
  - request_id_var      — ContextVar holding the current request's id
  - RequestIdFilter     — logging filter that copies request_id_var onto
                          every LogRecord as `req` (JSON formatter picks
                          this up automatically)
  - RequestContextMiddleware — FastAPI middleware that:
                          (a) reads incoming X-Request-Id or generates a
                              fresh UUID
                          (b) sets the contextvar for the request's
                              lifetime
                          (c) records start time + emits a structured
                              `[REQUEST]` log line on response with
                              method, path, status, duration_ms
                          (d) sets X-Request-Id on the outgoing response

Together these turn "grep timestamps and pray" into "filter @message
like /req=abc12345/" — one query reconstructs the full submit flow for
one user across access log → scorer log → DB log → response.
"""
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# Stored as a ContextVar so each FastAPI request (async task) sees its own
# value, even under concurrent load. Default empty string → JSON logs
# emit `req: ""` for log lines outside a request scope (background
# threads, startup hooks); easy to filter out.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    """Stamps the current request_id onto every LogRecord as `req`.

    JsonFormatter picks up any non-reserved LogRecord attribute and
    emits it as a top-level JSON field, so adding `req` here means
    every log line acquires a `"req": "abc12345"` field automatically.
    No changes needed at log.info() call sites.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Only set if not already present (some libraries set their own).
        if not hasattr(record, "req") or not record.req:
            record.req = request_id_var.get()
        return True


def _short_id() -> str:
    """8-char hex slug — readable enough for log lookup, long enough to
    avoid collision within a single 30-day log retention window."""
    return uuid.uuid4().hex[:8]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware: request_id + latency timing per request.

    - Honours an incoming X-Request-Id header (lets the frontend or an
      upstream proxy correlate). Falls back to a fresh 8-char id.
    - Sets the contextvar before downstream handlers run; resets on exit.
    - Emits ONE structured log line per request via the `request` logger:
        [REQUEST] method=POST path=/api/v1/... status=200 duration_ms=42
      Tagged via `extra={...}` so each field becomes a top-level JSON
      key — Logs Insights `stats avg(duration_ms) by path` works natively.
    """

    _log = logging.getLogger("request")

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("X-Request-Id")
        rid = incoming.strip()[:32] if incoming else _short_id()
        token = request_id_var.set(rid)
        started = time.monotonic()
        status = 500
        try:
            response: Response = await call_next(request)
            status = response.status_code
            # Echo the request id back so the frontend can include it in
            # support tickets / Sentry breadcrumbs / etc.
            response.headers["X-Request-Id"] = rid
            return response
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            # The `extra` dict lands as flat fields on the JSON log line.
            # Sticking to short field names so they're cheap to query.
            self._log.info(
                "[REQUEST] %s %s status=%d ms=%d",
                request.method, request.url.path, status, duration_ms,
                extra={
                    "method":      request.method,
                    "path":        request.url.path,
                    "status":      status,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)


def attach_request_id_to_response(response: Response) -> None:
    """Helper for handlers / middlewares that build their own Response
    objects before RequestContextMiddleware sees them. Most code paths
    don't need this — the middleware sets the header on the returned
    response automatically via dispatch. Kept for completeness."""
    rid = request_id_var.get()
    if rid:
        response.headers.setdefault("X-Request-Id", rid)
