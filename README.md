# AI Engineering con LLMs en Rust

> Serie educativa sobre implementación real de sistemas LLM desde los fundamentos hasta producción

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)]()

## ⚠️ Work in Progress

Este proyecto está en desarrollo activo. El roadmap puede cambiar según disponibilidad de tiempo e intereses personales o de la comunidad que decida contribuir. **Pull Requests son bienvenidos** para expandir la serie con nuevos conceptos o mejorar implementaciones existentes.

## 🎯 Sobre este Proyecto

Esta es una **serie de artículos técnicos** (no un curso formal) que explora la implementación de sistemas con Large Language Models desde una perspectiva pragmática y sin hype. El objetivo es entender y controlar cada capa del stack: desde tokenización hasta serving, pasando por inferencia, logits, sampling y RAG.

**Enfoque anti-hype**: Nada de "conecta LangChain y haz magia". Aquí construimos desde los fundamentos, entendiendo cómo funcionan realmente los LLMs y tomando control sobre cada componente del sistema.

**Por qué Rust**: Control de bajo nivel, rendimiento, safety y concurrencia. Ideal para entender los detalles de implementación sin sacrificar productividad.

## 📚 Contenido de la Serie

### 1. **Tokens** - Los Fundamentos
Entendiendo la tokenización: cómo el texto se convierte en números que un modelo puede procesar.

**Conceptos clave**:
- Tokenización con HuggingFace Tokenizers
- Vocabulario y encoding
- Manejo de caracteres especiales

**Ejecutar**:
```bash
make tokens
```

### 2. **Logits** - Entendiendo la Salida del Modelo
Análisis profundo de logits, probabilidades y sampling strategies.

**Conceptos clave**:
- Forward pass y generación de logits
- Conversión de logits a probabilidades (softmax)
- Estrategias de sampling (greedy, temperature-based)
- Carga y uso de modelos cuantizados (GGUF)
- Aceleración por hardware (Metal, CUDA)

**Ejecutar**:
```bash
make logits
```

### 3. **Haiku** - Generación de Texto Completa
Implementación end-to-end de un generador de texto con diferentes configuraciones.

**Conceptos clave**:
- Generación autoregresiva
- Control de temperatura y creatividad
- Manejo de tokens especiales (EOS)
- Streaming de output
- Comparación de estrategias de generación

**Ejecutar**:
```bash
make haiku
```

## 🚀 Quick Start

### Requisitos

- Rust 1.85+ (Edition 2024)
- Cargo
- Conexión a internet (para descargar modelos de HuggingFace)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/ai-engineering-rust.git
cd ai-engineering-rust

# Compilar todos los proyectos
cargo build --release

# Ejecutar cualquier ejemplo
make tokens
make logits
make haiku
```

### Configuración

El archivo `config.yaml` define los modelos y tokenizers utilizados:

```yaml
tokenizer:
  repo: "Qwen/Qwen3-4B"
  file: "tokenizer.json"

llm:
  repo: "unsloth/Qwen3-4B-GGUF"
  file: "Qwen3-4B-Q4_K_M.gguf"
  branch: "main"
```

Los modelos se descargan automáticamente desde HuggingFace Hub en la primera ejecución.

## 🏗️ Arquitectura del Proyecto

Este es un **workspace de Cargo** con múltiples crates independientes:

```
ai-engineering-rust/
├── tokens/          # Fundamentos de tokenización
├── logits/          # Análisis de logits y sampling
├── haiku/           # Generación de texto completa
├── config.yaml      # Configuración de modelos
└── Cargo.toml       # Workspace configuration
```

### Stack Tecnológico

- **[Candle](https://github.com/huggingface/candle)**: Framework de ML en Rust (HuggingFace)
- **[Tokenizers](https://github.com/huggingface/tokenizers)**: Tokenización rápida
- **[hf-hub](https://github.com/huggingface/hf-hub)**: Cliente para HuggingFace Hub
- **[GGUF](https://github.com/ggerganov/ggml)**: Formato de modelos cuantizados

### Aceleración por Hardware

El proyecto soporta múltiples backends de aceleración:

- **Metal**: Para GPUs de Apple Silicon
- **CUDA**: Para GPUs NVIDIA
- **Accelerate**: Framework de Apple para optimización en CPU
- **MKL**: Intel Math Kernel Library
- **CPU**: Fallback sin aceleración

## 🎓 Conceptos Explorados

### Tokenización
- Byte-Pair Encoding (BPE)
- Vocabulario y mapeo token-id
- Encoding y decoding
- Tokens especiales y control

### Inferencia
- Forward pass en transformers
- Carga de modelos cuantizados (Q4_K_M)
- Optimización de memoria con GGUF
- Detección y uso de aceleradores

### Logits y Probabilidades
- Raw logits vs probabilidades
- Softmax transformation
- Top-k analysis
- Interpretación de scores

### Sampling Strategies
- **Greedy**: Siempre el token más probable (temp ≈ 0)
- **Balanced**: Temperature moderada (0.7)
- **Creative**: Alta temperatura (1.5)
- **Chaos**: Temperatura muy alta (2.0)

### Generación Autoregresiva
- Loop de generación token-by-token
- Manejo de contexto posicional
- Early stopping con EOS tokens
- Streaming de output

## 🔧 Comandos Útiles

```bash
# Compilar todo el workspace
cargo build --release

# Ejecutar con features específicos (ejemplo: Metal en macOS)
cargo run --bin logits --features metal,accelerate

# Limpiar builds
cargo clean

# Verificar dependencias
cargo tree

# Ejecutar con logs detallados
RUST_LOG=debug make haiku
```

## 📖 Filosofía del Proyecto

### Pragmatismo sobre Hype
En lugar de usar abstracciones de alto nivel que ocultan la complejidad, este proyecto:
- Expone los detalles de implementación
- Explica el "por qué" de cada decisión técnica
- Muestra trade-offs reales (velocidad vs calidad, memoria vs precisión)
- No asume que "más complejo = mejor"

### Control Real
- Acceso directo a logits pre-softmax
- Implementación custom de sampling
- Manipulación explícita de tensores
- Sin capas de abstracción innecesarias

### Aprendizaje Profundo
No es suficiente con "hacer que funcione". El objetivo es:
- Entender cada componente del pipeline
- Poder debuggear problemas reales
- Tomar decisiones informadas sobre arquitectura
- Construir intuición sobre el comportamiento de los LLMs

## 🛣️ Roadmap

> **Nota**: Este roadmap es flexible y puede cambiar según disponibilidad de tiempo e intereses de la comunidad.

- [x] Tokenización básica
- [x] Inferencia y análisis de logits
- [x] Generación de texto con sampling
- [ ] Implementación de RAG (Retrieval-Augmented Generation)
- [ ] Embeddings y búsqueda semántica
- [ ] Serving con API REST
- [ ] Streaming de respuestas con SSE
- [ ] Fine-tuning con LoRA
- [ ] Evaluación y benchmarking

¿Tienes ideas para expandir la serie? **¡Los Pull Requests son bienvenidos!**

## 🤝 Contribuciones

Este es un proyecto educativo abierto y **los Pull Requests son bienvenidos**. Como este es un trabajo en progreso que evoluciona según disponibilidad de tiempo e intereses de la comunidad, tu participación puede ayudar a expandir y mejorar la serie.

### Cómo contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

### Guías de Contribución
- Mantén el enfoque pragmático y educativo
- Documenta decisiones técnicas y el "por qué"
- Incluye ejemplos ejecutables y reproducibles
- Evita abstracciones innecesarias
- Si añades un nuevo módulo, actualiza el README y el Makefile

### Ideas de Contribución
- Nuevos ejemplos explorando conceptos específicos
- Optimizaciones de rendimiento
- Soporte para nuevos modelos o arquitecturas
- Mejoras en documentación y explicaciones
- Herramientas de visualización o debugging

## 🙏 Agradecimientos

- [HuggingFace](https://huggingface.co/) por Candle y los modelos
- [Qwen Team](https://github.com/QwenLM) por Qwen3
- [unsloth](https://huggingface.co/unsloth) por las versiones GGUF optimizadas
- La comunidad de Rust por herramientas excepcionales

## 📚 Recursos Adicionales

- [Candle Examples](https://github.com/huggingface/candle/tree/main/candle-examples)
- [Tokenizers Docs](https://huggingface.co/docs/tokenizers/index)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Paper original de Transformers)

---

**Construido con 🦀 Rust y ❤️ por Luis Correa**
