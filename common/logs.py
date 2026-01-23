from __future__ import annotations

import logging
import logging.config
import os
from typing import Literal, assert_never, cast, get_args

LogLevel = Literal[
    "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"
]
LogFormat = Literal["text", "rich"]  # FIXME: Not really format

LOG_LEVELS: set[LogLevel] = set(get_args(LogLevel))
LOG_FORMATS: set[LogFormat] = set(get_args(LogFormat))


def validate_log_level(value: str | int | None) -> LogLevel:
    if isinstance(value, int):
        value = logging.getLevelName(value)

    if value not in LOG_LEVELS:
        choices = ", ".join(map(repr, LOG_LEVELS))
        msg = f"Invalid log level {value!r}, choices are {choices}"
        raise ValueError(msg)
    return cast("LogLevel", value)


def validate_log_format(value: str | None) -> LogFormat:
    if value not in LOG_FORMATS:
        choices = ", ".join(map(repr, LOG_FORMATS))
        msg = f"Invalid log format {value!r}, choices are {choices}"
        raise ValueError(msg)
    return cast("LogFormat", value)


def configure_logging(
    *namespaces: str,
    level: LogLevel | int | None = None,
    format_: LogFormat | None = None,
    capture_warnings: bool = True,
    root_level: LogLevel | int = "WARNING",
    disable_existing_loggers: bool = False,
    overwrite: bool = True,
) -> None:
    """Configure logging for the given namespaces.

    Parameters
    ----------
    level : LogLevel | int | None, optional
        Log level for the selected namespaces, by default None.
    format_ : LogFormat | None, optional
        Pre-configured logging formatter to use, by default None.
    capture_warnings : bool, optional
        Whether to capture warnings and log them or not, by default True.
    root_level : LogLevel | int, optional
        The default level for all other loggers, by default "WARNING".
    disable_existing_loggers : bool, optional
        Whether to disable existing loggers or not, by default False.
    overwrite : bool, optional
        Whether to overwrite existing configuration or not, by default True.
    """
    level = validate_log_level(level or os.getenv("LOG_LEVEL") or "INFO")
    root_level = validate_log_level(root_level)
    format_ = validate_log_format(format_ or os.getenv("LOG_FORMAT") or "text")
    # NOTE: Always add __main__
    namespaces = ("__main__", *namespaces)

    # NOTE: Could add process/thread info
    match format_:
        case "text":
            format_str = "%(asctime)s [%(name)s] %(levelname)-8s %(message)s"
            handler = "logging.StreamHandler"
        case "rich":
            format_str = "%(message)s"
            handler = "rich.logging.RichHandler"
        case never:
            assert_never(never)

    logging.captureWarnings(capture_warnings)
    logging.config.dictConfig(
        {
            "disable_existing_loggers": disable_existing_loggers,
            "incremental": not overwrite,
            "formatters": {
                "default": {
                    "format": format_str,
                }
            },
            "handlers": {
                "console": {
                    "class": handler,
                    "formatter": "default",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": root_level,
            },
            "loggers": {
                namespace: {"level": level, "propagate": True}
                for namespace in namespaces
            },
            "version": 1,
        }
    )
