from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

from src_bench.domain.enums import AdapterType, FrameworkType


@dataclass
class FrameworkBenchConfig:
    framework: FrameworkType
    batches_sizes: list[int]
    quantization_types: list[str]


@dataclass
class BenchConfig:
    measure_number: int
    adapter_type: AdapterType
    frameworks_config: list[FrameworkBenchConfig]


@dataclass
class FrameworkResult:
    framework: FrameworkType
    batch_size: int
    quantization: str
    execution_time: float
    image_size: Optional[str]


@dataclass
class BenchRunData:
    measure_number: int
    measure_function: Callable[[Iterable[str]], list[list[float]]]
    data: Sequence[str]
    batch_size: int
