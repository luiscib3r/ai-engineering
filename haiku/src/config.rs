use color_eyre::{Result, eyre::Ok};
use config::{Config, File, FileFormat};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub tokenizer: TokenizerConfig,
    pub llm: LlmConfig,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub repo: String,
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct LlmConfig {
    pub repo: String,
    pub file: String,
    pub branch: String,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config = Config::builder()
            .add_source(File::new("config", FileFormat::Yaml))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}
