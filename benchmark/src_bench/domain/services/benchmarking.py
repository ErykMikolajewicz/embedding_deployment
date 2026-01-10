from statistics import median

from src_bench.domain.models import BenchRunData
from src_bench.domain.services.measure import measure_execution_time


def benchmark_function(bench_run_data: BenchRunData) -> float:
    measure_numbers = bench_run_data.measure_number
    batch_size = bench_run_data.batch_size
    benchmark_data = bench_run_data.data
    measure_function = bench_run_data.measure_function

    measure_results = []
    for measure_number in range(measure_numbers):
        result = measure_execution_time(benchmark_data, batch_size, measure_function)
        result_number = result.seconds + result.microseconds / 1_000_000
        result = round(result_number, 2)
        measure_results.append(result)
    min_result = median(measure_results)

    return min_result
