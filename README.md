# AI Engineering con LLMs en Rust

> Serie educativa sobre implementación de sistemas LLM desde los fundamentos hasta producción

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)]()

## 🎯 Sobre este Proyecto

Serie de artículos técnicos que explora la implementación de sistemas con Large Language Models desde una perspectiva pragmática. El objetivo es entender y controlar cada capa del stack: desde tokenización hasta serving, pasando por inferencia, logits, sampling y RAG.

**Enfoque**: Sin abstracciones mágicas. Construimos desde los fundamentos, entendiendo cómo funcionan realmente los LLMs.

**Por qué Rust**: Control de bajo nivel, rendimiento, safety y concurrencia.

## 📝 Artículos

Serie de artículos técnicos que explican en profundidad los conceptos implementados en este repositorio:

1. [**Qué Pasa Cuando un LLM "Piensa": Tokens, Logits, y Sampling**](https://www.luisciber.com/p/que-pasa-cuando-un-llm-piensa-tokens)  
   Explicación completa del proceso interno de inferencia en LLMs: desde la tokenización del texto hasta la generación de respuestas, pasando por logits, probabilidades y estrategias de sampling.

## 📚 Contenido de la Serie

### 1. **Tokens** - Los Fundamentos
Entendiendo cómo el texto se convierte en números que un modelo puede procesar.

**Conceptos**: Tokenización con HuggingFace, vocabulario, encoding/decoding, caracteres especiales.

```bash
make tokens
```

### 2. **Logits** - Entendiendo la Salida del Modelo
Análisis de logits, probabilidades y estrategias de sampling.

**Conceptos**: Forward pass, logits a probabilidades (softmax), estrategias de sampling (greedy, temperature), modelos cuantizados (GGUF), aceleración por hardware.

```bash
make logits
```

### 3. **Haiku** - Generación de Texto Completa
Implementación end-to-end de un generador de texto con diferentes configuraciones.

**Conceptos**: Generación autoregresiva, control de temperatura, tokens especiales (EOS), streaming, comparación de estrategias.

```bash
make haiku
```

### 4. **LLM Inference** - Pipeline Completo de Inferencia
Implementación profesional del pipeline completo: prefill, KV cache, y loop autoregresivo.

**Conceptos**: Prefill optimizado, KV cache, generación autoregresiva eficiente, chat templates, configuración avanzada de sampling (top-k, top-p), estadísticas de generación.

```bash
make llm-inference
```

## 🚀 Quick Start

```bash
# Clonar el repositorio
git clone https://github.com/luisciber/ai-engineering-rust.git
cd ai-engineering-rust

# Compilar todos los proyectos
cargo build --release

# Ejecutar ejemplos
make tokens
make logits
make haiku
make llm-inference
```

### Configuración

El archivo `config.yaml` define los modelos y parámetros de inferencia:

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

## 🏗️ Arquitectura

Workspace de Cargo con múltiples crates independientes:

```
ai-engineering-rust/
├── tokens/          # Tokenización básica
├── logits/          # Análisis de logits y sampling
├── haiku/           # Generación de texto
├── llm-inference/   # Pipeline completo de inferencia
├── config.yaml      # Configuración de modelos
└── Cargo.toml       # Workspace configuration
```

### Stack Tecnológico

- **[Candle](https://github.com/huggingface/candle)**: Framework de ML en Rust (HuggingFace)
- **[Tokenizers](https://github.com/huggingface/tokenizers)**: Tokenización rápida
- **[hf-hub](https://github.com/huggingface/hf-hub)**: Cliente para HuggingFace Hub
- **[GGUF](https://github.com/ggerganov/ggml)**: Formato de modelos cuantizados

### Aceleración por Hardware

Soporta múltiples backends: Metal (Apple Silicon), CUDA (NVIDIA), Accelerate, MKL, CPU.

## 🎓 Conceptos Clave

- **Tokenización**: BPE, vocabulario, encoding/decoding
- **Inferencia**: Forward pass, modelos cuantizados, optimización de memoria
- **Logits**: Raw logits vs probabilidades, softmax, top-k analysis
- **Sampling**: Greedy, temperature-based, top-k, top-p
- **Generación**: Loop autoregresivo, KV cache, prefill, streaming, EOS tokens

## 🛣️ Roadmap

- [x] Tokenización básica
- [x] Inferencia y análisis de logits
- [x] Generación de texto con sampling
- [x] Pipeline completo de inferencia
- [ ] Implementación de RAG
- [ ] Embeddings y búsqueda semántica
- [ ] Serving con API REST
- [ ] Fine-tuning con LoRA

## 🤝 Contribuciones

Los Pull Requests son bienvenidos para expandir la serie con nuevos conceptos o mejorar implementaciones.

### Cómo contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

### Guías

- Mantén el enfoque pragmático y educativo
- Documenta decisiones técnicas y el "por qué"
- Incluye ejemplos ejecutables
- Evita abstracciones innecesarias

## 🙏 Agradecimientos

- [HuggingFace](https://huggingface.co/) por Candle y los modelos
- [Qwen Team](https://github.com/QwenLM) por Qwen3
- [unsloth](https://huggingface.co/unsloth) por las versiones GGUF optimizadas

## 📚 Recursos

- [Candle Examples](https://github.com/huggingface/candle/tree/main/candle-examples)
- [Tokenizers Docs](https://huggingface.co/docs/tokenizers/index)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

---

**Construido con 🦀 Rust y ❤️ por [Luis Correa](https://www.luisciber.com)**