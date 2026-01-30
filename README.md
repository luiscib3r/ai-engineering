# AI Engineering with LLMs in Rust

Educational series on implementing LLM systems from fundamentals to production.

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## About

A hands-on exploration of Large Language Models implementation. Each module builds understanding from the ground up: tokenization, inference, logits, sampling, and generation loops.

## Structure

```
ai-engineering-rust/
├── tokens/          # Basic tokenization
├── logits/          # Logits analysis and sampling
├── haiku/           # Text generation
├── llm-inference/   # Complete inference pipeline
├── config.yaml      # Model configuration
└── Cargo.toml       # Workspace
```

## Quick Start

```bash
git clone https://github.com/luisciber/ai-engineering-rust.git
cd ai-engineering-rust

# Build
cargo build --release

# Run examples
make tokens
make logits
make haiku
make llm-inference
```

## Modules

### 1. Tokens
Text-to-numbers conversion and tokenization basics.

```bash
make tokens
```

### 2. Logits
Model output analysis, probabilities, and sampling strategies.

```bash
make logits
```

### 3. Haiku
End-to-end text generation with configurable parameters.

```bash
make haiku
```

### 4. LLM Inference
Complete pipeline: prefill, KV cache, autoregressive loop.

```bash
make llm-inference
```

## Configuration

Edit `config.yaml` to customize models and inference parameters:

```yaml
tokenizer:
  repo: "Qwen/Qwen3-4B"
  file: "tokenizer.json"

llm:
  repo: "unsloth/Qwen3-4B-GGUF"
  file: "Qwen3-4B-Q4_K_M.gguf"

inference:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_length: 256
```

## Stack

- **[Candle](https://github.com/huggingface/candle)**: ML framework in Rust
- **[Tokenizers](https://github.com/huggingface/tokenizers)**: Fast tokenization
- **[hf-hub](https://github.com/huggingface/hf-hub)**: HuggingFace Hub client
- **[GGUF](https://github.com/ggerganov/ggml)**: Quantized model format

Supports Metal (Apple Silicon), CUDA, Accelerate, MKL, and CPU.

## Key Concepts

- **Tokenization**: BPE, vocabulary, encoding/decoding
- **Inference**: Forward pass, quantized models, memory optimization
- **Logits**: Raw logits, probabilities, softmax
- **Sampling**: Greedy, temperature, top-k, top-p
- **Generation**: Autoregressive loop, KV cache, prefill, streaming

## Contributing

Pull requests welcome. Keep it simple, practical, and well-documented.

1. Fork the repo
2. Create a branch
3. Make your changes
4. Submit a PR

## License

MIT

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for Candle and models
- [Qwen Team](https://github.com/QwenLM) for Qwen3
- [unsloth](https://huggingface.co/unsloth) for optimized GGUF versions