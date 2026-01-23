from __future__ import annotations

from typing import Annotated, ClassVar

import pydantic
import pydantic_settings


class DatabaseSettings(pydantic_settings.BaseSettings):
    database: Annotated[str, pydantic.Field(alias="db")] = "barometre"
    user: str = "user"
    host: str = "localhost"
    port: int = 5432
    password: pydantic.Secret[str] = pydantic.Secret("password")

    model_config: ClassVar[pydantic_settings.SettingsConfigDict] = {
        "env_prefix": "POSTGRES_",
        "env_nested_delimiter": "_",
        "extra": "forbid",
    }
