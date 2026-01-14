"""That encoder was written before was observed, that llama-cpp-python not work with gemma300m
It's actually functional, and waiting for time, when support for gemma will be added.
Issue of that problem on GitHub: https://github.com/abetlen/llama-cpp-python/issues/2065"""

# from llama_cpp import Llama
#
# from src.domain.quantization import Quantization
# from src.infrastructure.utils import get_model_root_path
#
#
# class LlamaCppEncoder:
#     def __init__(self, quantization: str):
#         self.__set_quantization(quantization)
#         self.__initialize_session()
#
#     def __initialize_session(self):
#         model_root = get_model_root_path()
#         model_path = f"{model_root}/llama_cpp/embeddinggemma-300M-{self.__quantization}.gguf"
#
#         self.__model = Llama(
#             model_path=model_path,
#             embedding=True,
#             n_ctx=2048,
#             n_batch=50,
#         )
#
#     def __set_quantization(self, quantization):
#         match quantization:
#             case Quantization.INT4:
#                 self.__quantization = "Q4_0"
#             case Quantization.INT8:
#                 self.__quantization = "Q8_0"
#             case Quantization.BF16:
#                 self.__quantization = "BF16"
#             case None:
#                 self.__quantization = "F32"
#             case _:
#                 raise Exception(f"Invalid quantization option {quantization}")
#
#     def encode(self, texts: list[str]) -> list[list[float]]:
#         embeddings = self.__model.embed(texts, normalize=True)
#
#         return embeddings
