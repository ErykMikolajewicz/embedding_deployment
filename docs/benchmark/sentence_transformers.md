Decent, but not great option.

The performance was good, better than ollama, but slightly iller than onnx.
Also, easy of use wasn't bad, for basic usage with not quantized model.

Unfortunately current quantization options are a bit of mess in PyTorch.

[Quantize dynamic](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html) is easy to use, but emit deprecation warnings, and have rather poor result, measured by quality of embeddings.

The successor of than option [torchao](https://docs.pytorch.org/ao/stable/index.html) look immature with limited docs and non-muteable deprecation warnings.
Also support for computing on cpu, and saving models for further usage is hard, and problematic.

The sentence-transformers is great for developing models, and training, but is not optimal deployment platform choose.

