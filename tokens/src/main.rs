use color_eyre::{Result, eyre::Error};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

mod config;

const HELLO_WORLD: &str = "Hello, world!";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    color_eyre::install()?;

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install crypto provider");

    let config = config::AppConfig::load()?;
    println!("\nUsing tokenizer: {}", config.tokenizer.repo);

    let api = hf_hub::api::tokio::Api::new()?;
    let tokenizer_path = api
        .model(config.tokenizer.repo)
        .get(&config.tokenizer.file)
        .await?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;

    let tokens_encoding = tokenizer
        .encode(HELLO_WORLD.to_string(), true)
        .map_err(Error::msg)?;

    let tokens = tokens_encoding.get_ids();

    println!("\nPrompt: {HELLO_WORLD}");
    println!("Tokens: {:?}", tokens);

    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocab size: {}", vocab_size);

    Ok(())
}
