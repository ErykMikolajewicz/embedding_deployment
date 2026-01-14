.venv/bin/python ./building/onnx/model_download.py --environment=local_test
.venv/bin/python ./building/onnx/model_download.py --quantization=fp16 --environment=local_test
.venv/bin/python ./building/onnx/model_download.py --quantization=int8 --environment=local_test
.venv/bin/python ./building/onnx/model_download.py --quantization=int4 --environment=local_test

.venv/bin/python ./building/sentence_transformers/model_download.py --environment=local_test

# GGUF download scripts are working, but llama-cpp-python currently (14.01.2026) not support gemma300M embedding
#.venv/bin/python ./building/llama_cpp/model_download.py --environment=local_test
#.venv/bin/python ./building/llama_cpp/model_download.py --quantization=bf16 --environment=local_test
#.venv/bin/python ./building/llama_cpp/model_download.py --quantization=int8 --environment=local_test
#.venv/bin/python ./building/llama_cpp/model_download.py --quantization=int4 --environment=local_test