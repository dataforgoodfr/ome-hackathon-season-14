from __future__ import annotations

from importlib.metadata import version

from ._logging import configure_logging

configure_logging("reporto")  # noqa: RUF067
__version__ = version("reporto")
