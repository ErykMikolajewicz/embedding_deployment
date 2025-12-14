from typing import Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.enums import QuantizationType, DeploymentOption

APPLICATION_SETTINGS_FILE_PATH = Path(".env")


class DeploymentSettings(BaseSettings):
    DEPLOYMENT_OPTION: DeploymentOption = ...

    QUANTIZATION: Optional[QuantizationType] = None

    model_config = SettingsConfigDict(
        env_file=APPLICATION_SETTINGS_FILE_PATH, env_file_encoding="utf-8", case_sensitive=True, frozen=True
    )

deployment_settings = DeploymentSettings()
