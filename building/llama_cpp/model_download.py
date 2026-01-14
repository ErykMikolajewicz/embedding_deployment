import argparse
from functools import partial
from pathlib import Path

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()

parser.add_argument("--quantization", required=False, nargs="?", default=None)

parser.add_argument("--environment", required=False, nargs="?", default="container")

args = parser.parse_args()
quantization = args.quantization
environment = args.environment

match environment:
    case "container":
        model_root_dir = "/embedding_deployment/llama_cpp"
    case "local_test":
        model_root_dir = "./models/llama_cpp"
    case _:
        raise Exception("Invalid environment!")

MODEL_ID = "unsloth/embeddinggemma-300m-GGUF"

local_dir = Path(model_root_dir)
download_model_file = partial(
    hf_hub_download, MODEL_ID, local_dir=local_dir
)

match quantization:
    case "int4":
        # fo this model naming is inconsistent, small "m" despite "M", like in other models
        int4_model_path = download_model_file(filename=f"embeddinggemma-300m-Q4_0.gguf")
        int4_model_path = Path(int4_model_path)
        int4_model_path.rename(local_dir / "embeddinggemma-300M-Q4_0.gguf")
    case "int8":
        quantization = "Q8_0"
        download_model_file(filename=f"embeddinggemma-300M-{quantization}.gguf")
    case "bf16":
        quantization = 'BF16'
        download_model_file(filename=f"embeddinggemma-300M-{quantization}.gguf")
    case None:
        quantization = 'F32'
        download_model_file(filename=f"embeddinggemma-300M-{quantization}.gguf")
    case _:
        raise Exception("Not supported quantization type!")
