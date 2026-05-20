"""Shared logging helpers for Coruscant consumers."""

from __future__ import annotations

import logging

from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def build_logger(
    name: str,
    log_path: Path,
    logger_root: str = "coruscant",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a dedicated file-and-stream logger for one task.

    Args:
            name: Short logical logger name for the current task.
            log_path: Path to the log file written by the logger.
            logger_root: Root namespace prepended to the logger name.
            level: Logging level applied to the logger and its handlers.

    Returns:
            A configured logger with stream and file handlers.
    """
    resolved_path = Path(log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{logger_root}.{name}.{resolved_path}")
    logger.setLevel(level)
    logger.propagate = False

    while logger.handlers:
        handler = logger.handlers.pop()
        handler.close()

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(resolved_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def logprint(
    msg: str,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Emit one message through a logger or standard output.

    Args:
            msg: Message text to emit.
            logger: Optional logger receiving the message.
            level: Logging level used when a logger is provided.

    Returns:
            None.
    """
    if logger is None:
        print(msg)
        return
    logger.log(level, msg)
