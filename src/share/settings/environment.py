from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.enums import Environment

ENV_FILE = Path(".env")


class EnvironmentSettings(BaseSettings):
    ENVIRONMENT: Environment = ...

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, case_sensitive=True, frozen=True, extra="ignore"
    )