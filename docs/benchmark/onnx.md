As a time of written, when only onnx, ollama, and sentence-transformers all on CPU are measured, the onnx is a clear winner.

The onnx cleary outperform both ollama and sentence transformers. The speedup is high, for f32 it is
~35% comparable to sentence-transformers, and near 100 % compared to ollama (available, as bf16).

For int8 where onnx was able to use effective quantization gains are even bigger.

Also, onnx was extremely easy to use, with quantization just working out of the box.
ollama was also working nice, by onnx have better support for quantization in my opinion.

As well the image size is very satisfying, 1.6GB vs 8.74GB for sentence_transformers, and 4,58GB for ollama.
