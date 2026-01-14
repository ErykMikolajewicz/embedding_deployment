from collections.abc import Callable, Sequence
from statistics import median

from src_bench.domain.services.measure import measure_execution_time


class BenchmarkRunner:
    def __init__(self, measure_number: int, benchmark_data: Sequence):
        self.__measure_number = measure_number
        self.__benchmark_data = benchmark_data

    def benchmark_function(self, measure_function: Callable, batch_size: int) -> float:
        measure_results = []
        for measure_number in range(self.__measure_number):
            result = measure_execution_time(self.__benchmark_data, batch_size, measure_function)
            result_number = result.seconds + result.microseconds / 1_000_000
            result = round(result_number, 2)
            measure_results.append(result)
        median_time = median(measure_results)

        return median_time
