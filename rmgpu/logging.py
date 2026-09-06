"""Central logging configuration for rmgpu.

Provides a module-level logger and a setup helper that can be invoked from the CLI
or from the driver `run()` entry point. The default format is

    2026-09-04 12:34:56,789 INFO  rmgpu.module message

The logger is configured once per process. Subsequent calls to setup_logging are
idempotent – the level and handlers are updated in place.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_log = logging.getLogger("rmgpu")
_log.setLevel(logging.INFO)  # default, overridden by setup_logging


def setup_logging(level: int | str = logging.INFO, log_file: Optional[str | Path] = None, *, force: bool = False) -> None:
    """Configure the rmgpu logger.

    Args:
        level: logging level as int or name (DEBUG/INFO/...).
        log_file: optional path to a file that receives the same records as stderr.
        force: if True, remove existing handlers before adding new ones.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if force:
        for h in list(_log.handlers):
            _log.removeHandler(h)

    if not _log.handlers:
        fmt = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=datefmt)

        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(formatter)
        _log.addHandler(stream)

    # Update level
    _log.setLevel(level)
    for h in _log.handlers:
        h.setLevel(level)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        fmt = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        file_handler.setLevel(level)
        # Avoid duplicate file handlers on repeated calls
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path) for h in _log.handlers):
            _log.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for the given module name."""
    return logging.getLogger(f"rmgpu.{name}")


# Expose a module-level logger for direct use
log = _log
