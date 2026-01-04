.venv/bin/python ./building/onnx/model_download.py --quantization=fp16 --environment=local_test
.venv/bin/python ./building/onnx/model_download.py --quantization=int8 --environment=local_test
.venv/bin/python ./building/onnx/model_download.py --quantization=int4 --environment=local_test
.venv/bin/python ./building/sentence_transformers/model_download.py --environment=local_test
.venv/bin/python ./building/sentence_transformers/model_download.py --quantization=int8 --environment=local_test
.venv/bin/python ./building/sentence_transformers/model_download.py --quantization=int4 --environment=local_test