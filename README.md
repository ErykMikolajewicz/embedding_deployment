# Embedding Model Deployment Benchmark

## Project description

The project was created out of the need to find a more efficient way to build container images with models for generating embeddings. The previous approach based on **Hugging Face** and **sentence-transformers**, although convenient to use, resulted in container images of **unacceptably large size** (around 12 GB).

The main cause of this issue was the necessity to include machine learning libraries such as:
- PyTorch  
- Triton

## Project Goal

The goal of the project is to:
- test different methods of building container images,
- reduce image size,
- compare the performance of different environments and deployment approaches.

## Model Used

As part of the project, the **gemma300m** embedding model is used.

This model was selected due to:
- its **versatility**, not only for semantic search but also for tasks such as clustering,
- good results in benchmarks,
- a relatively **small size**, enabling reasonable deployment on:
  - CPU,
  - GPU.

## Current Project Status

Currently prepared:
- scripts for building container images,
- scripts for deploying the model using CPU on:
  - **Ollama**,
  - **ONNX Runtime**,
  - **sentence transformers**.

### Model Quantization

The image build process allows selecting different quantization variants, for example:
- `int8`,
- `fp16`.
Available options depend strongly on used framework. 

## Development Plans

In the next stages of the project, the following are planned:
- comparison of the **performance**, and **accuracy** of different deployment options for embedding model,
- testing of additional environments, including:
  - `llama.cpp`,
- deployment of models on **GPU**.

## Possible Extensions

It is also being considered to extend the project with:
- deployment in the **Google Cloud Platform (GCP)** environment