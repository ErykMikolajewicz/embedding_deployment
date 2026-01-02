import tempfile, shutil
import os
import argparse

os.environ['HF_HOME']='/embedding_deployment/hf_home'
os.environ['HF_HUB_CACHE']='/embedding_deployment/hf_home/hub'

from huggingface_hub import snapshot_download
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from torch.ao.quantization import quantize_dynamic
import torch
import torch.nn as nn

token = open('/run/secrets/hf_token').read().strip()
login(token)
repo_id = "google/embeddinggemma-300m"

local_dir = "/embedding_deployment/models/embeddinggemma-300m"
tmp_cache = tempfile.mkdtemp(prefix="hf_cache_")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    cache_dir=tmp_cache,
    local_dir_use_symlinks=False,
)

shutil.rmtree(tmp_cache)

parser = argparse.ArgumentParser()
parser.add_argument("--quantization",
                    required=False,
                    nargs="?",
                    default=None)
args = parser.parse_args()

quantization = args.quantization

if quantization:
    match quantization:
        case 'int8':
            dtype = torch.qint8
        case 'f16':
            dtype = torch.float16
        case _:
            raise Exception('Invalid quantization!')

    model = SentenceTransformer(local_dir, device="cpu")

    quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
        inplace=True
    )
