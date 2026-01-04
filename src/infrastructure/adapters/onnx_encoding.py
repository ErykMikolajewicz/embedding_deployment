from collections.abc import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.infrastructure.utils.paths import get_model_root_path
from src.share.settings.quantization import Quantization, quantization_settings

model_root = get_model_root_path()

match quantization_settings.QUANTIZATION:
    case Quantization.INT4:
        quantization = "_q4"
    case Quantization.INT8:
        quantization = "_int8"
    case Quantization.FP16:
        quantization = "_fp16"
    case None:
        quantization = ""
    case _:
        raise Exception(f"Invalid quantization option {quantization_settings.QUANTIZATION}")

model_path = f"{model_root}/onnx/model{quantization}.onnx"

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

tokenizer = Tokenizer.from_file(f"{model_root}/onnx/tokenizer/tokenizer.json")
tokenizer.enable_padding(length=None, pad_id=0)


def encode(texts: Iterable[str]) -> list[list[float]]:
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    _, embeddings = session.run(None, onnx_inputs)

    embeddings = embeddings.tolist()

    return embeddings
