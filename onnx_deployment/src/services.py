import onnxruntime as ort
from transformers import AutoTokenizer

model_path = '/embedding_deployment/onnx/model.onnx'

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

tokenizer = AutoTokenizer.from_pretrained("/embedding_deployment/onnx/tokenizer")

def encode(texts: list[str]) -> list[list[float]]:

    encoded_texts = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
    )

    input_names = [input_params.name for input_params in session.get_inputs()]
    onnx_inputs = {name: encoded_texts[name] for name in input_names}

    _, embeddings = session.run(None, onnx_inputs)

    embeddings = embeddings.tolist()

    return embeddings