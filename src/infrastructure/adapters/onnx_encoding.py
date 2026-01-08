from collections.abc import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.domain.quantization import Quantization
from src.share.consts import MODEL_ROOT


class OnnxEncoder:
    def __init__(self, quantization: str):
        self.__set_quantization(quantization)
        self.__initialize_session()

    def __initialize_session(self):
        model_path = f"{MODEL_ROOT}/onnx/model{self.__quantization}.onnx"
        self.__session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.__tokenizer = Tokenizer.from_file(f"{MODEL_ROOT}/onnx/tokenizer/tokenizer.json")
        self.__tokenizer.enable_padding(length=None, pad_id=0)

    def __set_quantization(self, quantization):
        match quantization:
            case Quantization.INT4:
                self.__quantization = "_q4"
            case Quantization.INT8:
                self.__quantization = "_int8"
            case Quantization.FP16:
                self.__quantization = "_fp16"
            case None:
                self.__quantization = ""
            case _:
                raise Exception(f"Invalid quantization option {quantization}")

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        encodings = self.__tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        _, embeddings = self.__session.run(None, onnx_inputs)

        embeddings = embeddings.tolist()

        return embeddings
