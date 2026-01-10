# Benchmark of the **gemma-300m** Embedding Model

As part of the benchmark, embeddings were generated for articles of Polish law act "Kodeks postępowania administracyjnego".
The articles were sorted by length in ascending order.
For each batch size, the entire Code of Administrative Procedure was processed.

## Test Environment

- **Processor:** 13th Gen Intel Core i7-13650HX
- **Hyperthreading:** disabled, this is how it happened to be configured on the machine
- **Operating System:** Linux
- **Hardware support:**
  - available CPU flags enabling efficient **INT8** integer computations  
  - no dedicated hardware support for **FP16**; the CPU casts to **F32**
  - no flags indicating support for other types, e.g. **INT4**

## Methodology

- Number of measurements per case: **3**
- The lowest result from each sample was selected
- Measured parameter: **execution time**

## Benchmark Results

### Execution Time Rest
| Runtime | Batch size | F32 [s] | FP16 [s] | INT8 [s] |
|---------|------------|---------|----------|----------|
| ONNX    | 5          | 19.71   | 19.35    | 10.05    |
| ONNX    | 10         | 22.23   | 23.93    | 11.47    |
| ONNX    | 20         | 25.21   | 27.48    | 12.99    |
| ONNX    | 50         | 35.88   | 39.37    | 19.23    |
| ollama  | 5          | -       | 39.71    | 39.86    |
| ollama  | 10         | -       | 41.70    | 41.16    |
| ollama  | 20         | -       | 41.74    | 40.99    |
| ollama  | 50         | -       | 42.27    | 40.25    |


### Execution Time Direct
| Runtime               | Batch size | F32 [s] | BF16 [s] | FP16 [s] | INT8 [s] | INT4 [s] |
|-----------------------|------------|---------|----------|----------|----------|----------|
| ONNX                  | 5          | 21.65   | -        | 23.87    | 11.88    | 22.53    |
| ONNX                  | 10         | 22.53   | -        | 24.76    | 11.55    | 23.15    |
| ONNX                  | 20         | 24.97   | -        | 27.77    | 13.08    | 26.11    |
| ONNX                  | 50         | 35.54   | -        | 39.56    | 19.85    | 35.62    |
| sentence_transformers | 5          | 30.30   | 116.99   | -        | -        | -        |
| sentence_transformers | 10         | 29.36   | 120.23   | -        | -        | -        |
| sentence_transformers | 20         | 34.25   | 132.51   | -        | -        | -        |
| sentence_transformers | 50         | 51.7    | 206.36   | -        | -        | -        |

### Model Size
| Runtime | F32    | FP16   | INT8   |
|---------|--------|--------|--------|
| ONNX    | 1.59GB | 973MB  | 668MB  |
| ollama  | -      | 4.57GB | 4.29GB |

## Conclusions

### ONNX
- A clear batching effect is visible. For smaller batch sizes, execution was noticeably faster.
- **INT8** is clearly the fastest, as expected.
- **FP16** is slightly slower than **F32**; the lack of hardware support likely forces casting to **F32** on the CPU.

### Ollama
- Clearly slower than ONNX in every configuration; the results are decidedly disappointing.
- Only a minor benefit from smaller batch sizes; this parameter has little impact.
- It was not possible to take advantage of efficient INT8 instructions, or the benefits were overshadowed by other overheads.

**Summary:**
With ONNX, it was possible to significantly reduce the model size while achieving good performance, especially for INT8.
The situation is worse with Ollama: the model image remained relatively large over 4GB and performance was relatively low.
The only advantage of using Ollama is its relative simplicity, which unfortunately comes at the cost of a significant loss of flexibility.
