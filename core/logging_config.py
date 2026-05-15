"""Structured logging setup for the backend.

Set ``LOG_LEVEL`` (default ``INFO``) and ``LOG_JSON`` (default ``true``) via
environment variables. JSON output is one record per line, suitable for
CloudWatch / Datadog / Loki ingestion. Set ``LOG_JSON=false`` for human-readable
text output during local development.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_JSON_LOGS = os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes"}

# Standard fields built into LogRecord; everything else is treated as extra.
_RESERVED_RECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging() -> None:
    """Configure root logger. Idempotent — safe to call multiple times."""
    root = logging.getLogger()
    if getattr(root, "_englishfirm_configured", False):
        return
    root.setLevel(_LOG_LEVEL)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if _JSON_LOGS else logging.Formatter(_TEXT_FORMAT))
    # Attach the request-id filter so every emitted record carries the
    # current request's id as the `req` field (when inside a request
    # scope). Importing here keeps the module import-cycle clean —
    # request_context imports starlette which doesn't import this module.
    from core.request_context import RequestIdFilter
    handler.addFilter(RequestIdFilter())
    root.handlers.clear()
    root.addHandler(handler)
    # Tame chatty third-party loggers.
    logging.getLogger("uvicorn.access").setLevel("WARNING")
    logging.getLogger("botocore").setLevel("WARNING")
    logging.getLogger("urllib3").setLevel("WARNING")
    root._englishfirm_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
