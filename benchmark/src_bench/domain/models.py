from dataclasses import dataclass

from src_bench.domain.enums import FrameworkType


@dataclass
class FrameworkBenchConfig:
    framework: FrameworkType
    batches_sizes: list[int]
    quantization_types: list[str]


@dataclass
class FrameworkResult:
    framework: FrameworkType
    batch_size: int
    quantization: str
    execution_time: float
    image_size: str
