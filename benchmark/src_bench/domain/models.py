from dataclasses import dataclass


@dataclass
class FrameworkBenchConfig:
    framework: str
    batches_sizes: list[int]
    quantization_types: list[str]