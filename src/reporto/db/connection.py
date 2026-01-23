from __future__ import annotations

import logging

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .settings import DatabaseSettings

_logger = logging.getLogger(__name__)


def connect_to_db(settings: DatabaseSettings | None = None) -> Engine:
    """Connect to the PostgreSQL database using given or env variable settings."""
    if settings is None:
        settings = DatabaseSettings()

    _logger.info("Connecting to %s for DB %s", settings.host, settings.database)
    url = URL.create(
        drivername="postgresql",
        username=settings.user,
        host=settings.host,
        database=settings.database,
        port=settings.port,
        password=settings.password.get_secret_value(),
    )

    return create_engine(url)


def get_db_session(engine: Engine | None = None) -> Session:
    """
    Create a session for interacting with the database using the provided engine.

    Parameters
    ----------
    - engine (Engine, optional): The SQLAlchemy engine object. If not provided, it calls `connect_to_db()` to obtain one.

    Returns
    -------
    - Session: A SQLAlchemy session bound to the provided engine or created by calling `connect_to_db()`.
    """  # noqa: E501
    if engine is None:
        engine = connect_to_db()

    session = sessionmaker(bind=engine)
    return session()
