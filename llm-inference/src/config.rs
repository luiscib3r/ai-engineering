// src/config.rs
use color_eyre::Result;
use config::{Config, File, FileFormat};
use serde::Deserialize;

/// Configuración completa de la aplicación
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub tokenizer: TokenizerConfig,
    pub llm: LlmConfig,
    pub inference: InferenceConfig,
}

/// Configuración del tokenizer
#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub repo: String,
    pub file: String,
    pub eos_token: String,
    pub chat_template: String,
}

/// Configuración del modelo
#[derive(Debug, Deserialize)]
pub struct LlmConfig {
    pub repo: String,
    pub file: String,
    pub branch: String,
}

/// Cómo generar texto
#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub max_length: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

impl AppConfig {
    /// Cargar configuración desde config.yaml
    pub fn load() -> Result<Self> {
        let config = Config::builder()
            .add_source(File::new("config", FileFormat::Yaml))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}
