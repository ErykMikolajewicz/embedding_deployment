from typing import Iterable

from dishka import Provider, Scope, from_context, make_container, provide

from src.infrastructure.exceptions import AdapterNotSupported
from src_bench.config import BenchConfig, get_benchmark_config
from src_bench.domain.enums import AdapterType, FrameworkType
from src_bench.domain.ports import DirectAdapter, EnvironmentPreparator, MeasureFunction, RestAdapter
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
    match framework_type:
        case FrameworkType.OLLAMA:
            return OllamaAdapter
        case FrameworkType.ONNX:
            return CustomRestAdapter
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return CustomRestAdapter


def get_container_instantiate(framework_type: FrameworkType) -> type[EnvironmentPreparator]:
    match framework_type:
        case FrameworkType.OLLAMA:
            return OllamaContainerInstantiate
        case FrameworkType.ONNX:
            return OnnxContainerInstantiate
        case FrameworkType.SENTENCE_TRANSFORMERS:
            return SentenceTransformersContainerInstantiate


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
    ) -> Iterable[MeasureFunction]:
        match adapter_type:
            case AdapterType.DIRECT:
                direct_adapter_class = get_direct_adapter_class(framework_type)
                adapter = direct_adapter_class(quantization)
                yield adapter.get_embeddings
            case AdapterType.REST:
                rest_adapter_class = get_rest_adapter_class(framework_type)
                adapter = rest_adapter_class(port, quantization)

                container_instantiate = get_container_instantiate(framework_type)
                with container_instantiate(port, quantization):
                    yield adapter.get_embeddings


class MeasurerProvider(Provider):
    scope = Scope.SESSION

    @provide(scope=Scope.SESSION)
    def measurer(self, config: BenchConfig, function: MeasureFunction) -> Measurer:
        measure_number = config.measure_number
        benchmark_data = get_benchmark_data()
        return Measurer(measure_number, benchmark_data, function)


container = make_container(Config(), Function(), MeasurerProvider())
