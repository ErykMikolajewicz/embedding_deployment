from dataclasses import dataclass

from src_bench.domain.enums import FrameworkType, Quantization


@dataclass
class FrameworkResult:
    framework: FrameworkType
    batch_size: int
    quantization: Quantization
    execution_time: float
