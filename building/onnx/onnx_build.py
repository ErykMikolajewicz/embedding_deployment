from pathlib import Path
from functools import partial
import argparse

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--quantization",
                    required=False,
                    nargs="?",
                    default=None)
args = parser.parse_args()

quantization = args.quantization

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained("/embedding_deployment/onnx/tokenizer")

local_dir = Path('/embedding_deployment')
download_model_file = partial(hf_hub_download, MODEL_ID, subfolder="onnx", local_dir=local_dir)

def download_model(model_type = '', prefix = '') -> Path:
    model_path = download_model_file(filename=f"model{prefix}{model_type}.onnx")
    download_model_file(filename=f"model{prefix}{model_type}.onnx_data")
    return Path(model_path)

def quantize_dynamically(weight_type: QuantType):
    base_model_path = download_model()
    preprocess.quant_pre_process(base_model_path, base_model_path)

    quantize_dynamic(
        model_input=base_model_path,
        model_output=base_model_path,
        weight_type=weight_type
    )

    model_dir = Path(base_model_path.parent.name)
    old_data = model_dir / "model.onnx_data"
    old_data.unlink()


def download_and_rename(model_type: str):
    model_path = download_model(model_type, '_')

    model_dir = Path(model_path.parent.name)

    main_model_file = model_dir / f"model_{model_type}.onnx"
    main_model_file.rename(model_dir / "model.onnx")


if quantization:
    match quantization:
        case 'int4':
            quantization = 'q4'
            download_and_rename(quantization)
        case 'int8':
            quantize_dynamically(QuantType.QInt8)
        case 'fp16':
            download_and_rename(quantization)
        case _:
            raise Exception('Not supported quantization type!')
else:
    download_model()
