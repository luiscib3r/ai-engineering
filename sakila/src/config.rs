use color_eyre::Result;
use config::{Config, File, FileFormat};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub db: DbConfig,
    pub tokenizer: TokenizerConfig,
    pub llm: LlmConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DbConfig {
    pub url: String,
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub repo: String,
    pub file: String,
    pub eos_token: String,
    pub user_template: String,
    pub assistant_template: String,
    pub start_completion: String,
}

#[derive(Debug, Deserialize)]
pub struct LlmConfig {
    pub repo: String,
    pub file: String,
    pub branch: String,
}

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub max_length: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config = Config::builder()
            .add_source(File::new("config", FileFormat::Yaml))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}
