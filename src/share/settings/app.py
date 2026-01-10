from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.enums import DecoderType

ENV_FILE = Path(".env")


class AppSettings(BaseSettings):
    DECODER_TYPE: DecoderType = ...

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, case_sensitive=True, frozen=True, env_prefix="APP_", extra="ignore"
    )
