import tempfile, shutil
import os

os.environ['HF_HOME']='/embedding_deployment/hf_home'
os.environ['HF_HUB_CACHE']='/embedding_deployment/hf_home/hub'

from huggingface_hub import snapshot_download
from huggingface_hub import login

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
