import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

model_path = "/embedding_deployment/onnx/model.onnx"

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

tokenizer = Tokenizer.from_file("/embedding_deployment/onnx/tokenizer/tokenizer.json")
tokenizer.enable_padding(length=None, pad_id=0)


def encode(texts: list[str]) -> list[list[float]]:
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    _, embeddings = session.run(None, onnx_inputs)

    embeddings = embeddings.tolist()

    return embeddings
