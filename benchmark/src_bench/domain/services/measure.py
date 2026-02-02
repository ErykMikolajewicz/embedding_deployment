from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from statistics import median

from src_bench.domain.types import Embeddings


class Measurer:
    def __init__(self, measure_number: int, benchmark_data: Sequence, function: Callable[[Sequence[str]], Embeddings]):
        self.__measure_number = measure_number
        self.__benchmark_data = benchmark_data
        self.__function = function

    def measure_median_time(self, batch_size: int) -> float:
        measure_results = []
        for _ in range(self.__measure_number):
            result = self.measure_execution_time(batch_size)
            result_number = result.seconds + result.microseconds / 1_000_000
            result = round(result_number, 2)
            measure_results.append(result)
        median_time = median(measure_results)

        return median_time

    def measure_execution_time(self, batch_size: int) -> timedelta:
        num_articles = len(self.__benchmark_data)
        t1 = datetime.now()
        for i in range(0, num_articles, batch_size):
            batch_articles = self.__benchmark_data[i : i + batch_size]
            self.__function(batch_articles)

        t2 = datetime.now()
        return t2 - t1
