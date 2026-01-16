import argparse
from functools import partial
from pathlib import Path

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, preprocess, quantize_dynamic
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--quantization", required=False, nargs="?", default=None)

parser.add_argument("--environment", required=False, nargs="?", default="container")

args = parser.parse_args()
quantization = args.quantization
environment = args.environment

match environment:
    case "container":
        model_root_dir = "/embedding_deployment"
    case "local_test":
        model_root_dir = "./models"
    case _:
        raise Exception("Invalid environment!")

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(f"{model_root_dir}/onnx/tokenizer")

local_dir = Path(model_root_dir)
download_model_file = partial(hf_hub_download, MODEL_ID, subfolder="onnx", local_dir=local_dir)


def download_model(model_type="", prefix="") -> Path:
    model_path = download_model_file(filename=f"model{prefix}{model_type}.onnx")
    download_model_file(filename=f"model{prefix}{model_type}.onnx_data")
    return Path(model_path)


def quantize_dynamically(weight_type: QuantType):
    base_model_path = download_model()
    preprocess.quant_pre_process(base_model_path, base_model_path)

    new_model_path = f"{model_root_dir}/onnx/model_{quantization}.onnx"
    quantize_dynamic(
        model_input=base_model_path,
        model_output=new_model_path,
        weight_type=weight_type,
    )

    match environment:
        case "container":
            base_model_path.unlink()
            model_dir = Path(base_model_path.parent.name)
            old_data = model_dir / "model.onnx_data"
            old_data.unlink()


if quantization:
    match quantization:
        case "int4":
            quantization = "q4"
            download_model(quantization, "_")
        case "int8":
            quantize_dynamically(QuantType.QInt8)
        case "fp16":
            download_model(quantization, "_")
        case None:
            download_model()
        case _:
            raise Exception("Not supported quantization type!")
else:
    download_model()
