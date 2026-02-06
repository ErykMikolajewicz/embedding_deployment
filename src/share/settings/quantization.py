from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.enums import Quantization

ENV_FILE = Path(".env")


class QuantizationSettings(BaseSettings):
    QUANTIZATION: Optional[Quantization] = None

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=True, frozen=True, extra="ignore")


quantization_settings = QuantizationSettings()
