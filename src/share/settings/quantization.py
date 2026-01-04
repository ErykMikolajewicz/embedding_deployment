from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(".env")


class Quantization(StrEnum):
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"


class QuantizationSettings(BaseSettings):
    QUANTIZATION: Optional[Quantization] = None

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=True, frozen=True, extra="ignore")


quantization_settings = QuantizationSettings()
