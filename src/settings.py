from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.enums import DecoderType

ENV_FILE = Path(".env")

class Settings(BaseSettings):
    DECODER_TYPE: DecoderType = ...

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=True, frozen=True)


settings = Settings()
