"""
ASGI middleware stack:

1. RequestIDMiddleware  — injects X-Request-ID into every request/response.
   Enables distributed tracing: log the ID, grep for it, see the full story.

2. LoggingMiddleware    — structured JSON log per request with method, path,
   status, latency, and request_id. Easy to pipe into any log aggregator.

3. PrometheusMiddleware — records request count + latency histograms for every
   route. /metrics is excluded to avoid self-measurement noise.
"""

import json
import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from api.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger("fraud_api")


def configure_logging():
    """Emit structured JSON to stdout — forward to any aggregator (Datadog, Loki…)."""
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if hasattr(record, "request_id"):
                payload["request_id"] = record.request_id
            if hasattr(record, "extra"):
                payload.update(record.extra)
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request and echo it in the response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured log line per completed request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000

        request_id = getattr(request.state, "request_id", "-")
        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
        }
        record = logging.LogRecord(
            name="fraud_api", level=logging.INFO, pathname="", lineno=0,
            msg="request", args=(), exc_info=None,
        )
        record.__dict__.update({"extra": extra, "request_id": request_id})
        logger.handle(record)

        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency; skip /metrics to avoid self-loop."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path == "/metrics":
            return await call_next(request)

        t0 = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - t0

        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(path=path).observe(latency)

        return response
