import os
from collections.abc import Iterable, Sequence

from dishka import Provider, Scope, decorate, from_context, make_container, provide

from src.infrastructure.exceptions import AdapterNotSupported
from src_bench.config import BenchConfig, get_benchmark_config
from src_bench.domain.enums import AdapterType, FrameworkType
from src_bench.domain.ports import DirectAdapter, MeasureFunction, RestAdapter
from src_bench.domain.services.data import get_benchmark_data
from src_bench.domain.services.measure import Measurer
from src_bench.infrastructure.adapters import (
    CustomRestAdapter,
    DirectOnnxAdapter,
    DirectSentenceTransformersAdapter,
    OllamaAdapter,
)
from src_bench.infrastructure.containers import (
    OllamaContainerInstantiate,
    OnnxContainerInstantiate,
    SentenceTransformersContainerInstantiate,
)


def get_direct_adapter_class(framework_type: FrameworkType) -> type[DirectAdapter]:
    match framework_type:
        case FrameworkType.OLLAMA:
            raise AdapterNotSupported("direct adapter", framework_type)
        case FrameworkType.ONNX:
            return DirectOnnxAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return DirectSentenceTransformersAdapter


def get_rest_adapter_class(framework_type: FrameworkType) -> type[RestAdapter]:
    os.environ["ENVIRONMENT"] = "LOCAL"
    match framework_type:
        case FrameworkType.OLLAMA:
            return OllamaAdapter
        case FrameworkType.ONNX:
            return CustomRestAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return CustomRestAdapter


class Config(Provider):
    @provide(scope=Scope.APP)
    def config(self) -> BenchConfig:
        return get_benchmark_config()

    @provide(scope=Scope.APP)
    def adapter_type(self, config: BenchConfig) -> AdapterType:
        return config.adapter_type

    @provide(scope=Scope.APP)
    def port(self, config: BenchConfig) -> int:
        return config.rest_port


class Function(Provider):
    scope = Scope.SESSION

    framework_type = from_context(provides=FrameworkType, scope=Scope.SESSION)
    quantization = from_context(provides=str, scope=Scope.SESSION)

    @provide(scope=Scope.SESSION)
    def measure_function(
        self, adapter_type: AdapterType, port: int, framework_type: FrameworkType, quantization: str
    ) -> MeasureFunction:
        match adapter_type:
            case AdapterType.DIRECT:
                direct_adapter_class = get_direct_adapter_class(framework_type)
                adapter = direct_adapter_class(quantization)
            case AdapterType.REST:
                rest_adapter_class = get_rest_adapter_class(framework_type)
                adapter = rest_adapter_class(port, quantization)
        return adapter.get_embeddings

    @decorate(scope=Scope.SESSION)
    def instantiate_container(
        self,
        measure_function: MeasureFunction,
        adapter_type: AdapterType,
        port: int,
        framework_type: FrameworkType,
        quantization: str,
    ) -> Iterable[MeasureFunction]:
        if adapter_type == AdapterType.DIRECT:
            yield measure_function
        else:
            match framework_type:
                case FrameworkType.OLLAMA:
                    container_instantiate = OllamaContainerInstantiate
                case FrameworkType.ONNX:
                    container_instantiate = OnnxContainerInstantiate
                case FrameworkType.SENTENCE_TRANSFORMERS:
                    container_instantiate = SentenceTransformersContainerInstantiate

            with container_instantiate(port, quantization):
                yield measure_function


class Data(Provider):
    @provide(scope=Scope.APP)
    def benchmark_data(self) -> Sequence[str]:
        benchmark_data = get_benchmark_data()
        return benchmark_data


class MeasurerProvider(Provider):
    scope = Scope.SESSION

    @provide(scope=Scope.SESSION)
    def measurer(self, config: BenchConfig, benchmark_data: Sequence[str], function: MeasureFunction) -> Measurer:
        measure_number = config.measure_number
        return Measurer(measure_number, benchmark_data, function)


container = make_container(Config(), Data(), Function(), MeasurerProvider())
