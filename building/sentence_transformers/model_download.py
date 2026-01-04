import tempfile, shutil
import os
import argparse

from huggingface_hub import snapshot_download
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--quantization", required=False, nargs="?", default=None)
parser.add_argument("--environment", required=False, nargs="?", default="container")
args = parser.parse_args()

environment = args.environment
if environment == 'container':
    os.environ['HF_HOME']='/embedding_deployment/hf_home'
    os.environ['HF_HUB_CACHE']='/embedding_deployment/hf_home/hub'
    with open('/run/secrets/hf_token', 'r') as f:
        token = f.read().strip()
else:
    with open('./secrets/hf_token.txt', 'r') as f:
        token = f.read().strip()
login(token)

repo_id = "google/embeddinggemma-300m"

match environment:
    case "container":
        model_root_dir = "/embedding_deployment"
    case "local_test":
        model_root_dir = "./models"
    case _:
        raise Exception("Invalid environment!")


quantization = args.quantization
if quantization is None:
    local_dir = f"{model_root_dir}/sentence_transformers/embeddinggemma-300m"
else:
    local_dir = tempfile.mkdtemp()
tmp_cache = tempfile.mkdtemp(prefix="hf_cache_")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    cache_dir=tmp_cache
)

shutil.rmtree(tmp_cache)

if quantization:
    from torchao.quantization import (quantize_, Int4DynamicActivationInt4WeightConfig,
                                      Int8DynamicActivationInt8WeightConfig)

    match quantization:
        case 'int4':
            quantization_config = Int4DynamicActivationInt4WeightConfig()
        case 'int8':
            quantization_config = Int8DynamicActivationInt8WeightConfig()
        case _:
            raise Exception('Invalid quantization!')

    model = SentenceTransformer(local_dir, device="cpu")

    quantize_(model, quantization_config)

    save_path = f"{model_root_dir}/sentence_transformers/embeddinggemma-300m_{quantization}"
    model.save(save_path, safe_serialization=False)

    shutil.rmtree(local_dir)
