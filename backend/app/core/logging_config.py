"""Structured (JSON) logging configuration.

Every future service should log through the standard ``logging`` module; calling
:func:`configure_logging` once at startup routes those records to stdout as
single-line JSON objects, which are trivially ingestible by log aggregators.
No ``print`` statements anywhere in the codebase.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys

from app.core.config import settings


class JsonFormatter(logging.Formatter):
    """Format log records as compact single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Surface any structured extras attached via logger.*(..., extra={...}).
        standard = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in standard and key != "message":
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure the root logger for JSON stdout output at the configured level."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL.upper())
