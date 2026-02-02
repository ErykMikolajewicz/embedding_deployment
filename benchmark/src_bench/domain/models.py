from collections.abc import Sequence
from dataclasses import dataclass

from src_bench.domain.enums import FrameworkType
from src_bench.domain.ports import MeasureFunction


@dataclass
class FrameworkResult:
    framework: FrameworkType
    batch_size: int
    quantization: str
    execution_time: float


@dataclass
class BenchRunData:
    measure_number: int
    measure_function: MeasureFunction
    data: Sequence[str]
    batch_size: int
