from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.enums import DecoderType

ENV_FILE = Path(".env")


class Environment(StrEnum):
    CONTAINER = "CONTAINER"
    LOCAL_TEST = "LOCAL_TEST"


class AppSettings(BaseSettings):
    DECODER_TYPE: DecoderType = ...
    ENVIRONMENT: Environment = ...

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, case_sensitive=True, frozen=True, env_prefix="APP_", extra="ignore"
    )


app_settings = AppSettings()
