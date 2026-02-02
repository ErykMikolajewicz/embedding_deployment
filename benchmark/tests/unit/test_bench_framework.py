# from unittest.mock import Mock
#
# from src_bench.domain.services.measure import Measurer
#
#
# def test_benchmark_framework(sentences):
#     measure_function = Mock(return_value=None)
#
#     benchmark_runner = Measurer(measure_number=5, benchmark_data=sentences)
#
#     benchmark_result = benchmark_runner.measure(measure_function, batch_size=50)
#
#     assert isinstance(benchmark_result, float)
