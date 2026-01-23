from __future__ import annotations

from typing import Annotated

import cyclopts

from reporto import __version__
from reporto._logging import LogFormat, LogLevel, configure_logging

from . import _run

__all__ = ["cli"]

_cli = cyclopts.App(
    name="reporto",
    version=__version__,
    help="Hackathon OMÉ x Data For Good on TV Report Classification",
)

_cli.meta.group_parameters = cyclopts.Group("Logging")


@_cli.meta.default
def _configure_logging(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_format: Annotated[
        LogFormat, cyclopts.Parameter(env_var="REPORTO__LOGGING__FORMAT")
    ] = "text",
    log_level: Annotated[
        LogLevel, cyclopts.Parameter(env_var="REPORTO__LOGGING__LEVEL")
    ] = "INFO",
) -> None:
    configure_logging(
        "reporto",
        format_=log_format,
        level=log_level,
        capture_warnings=True,
    )
    _cli(tokens)


# CAUTION: Using the meta app to have shared logging
# https://cyclopts.readthedocs.io/en/latest/meta_app.html#meta-sub-app
cli = _cli.meta

cli.command(_run.train)
cli.command(_run.predict)

if __name__ == "__main__":
    cli()
