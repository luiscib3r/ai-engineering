use candle_core::{Tensor, quantized::gguf_file};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use color_eyre::{Result, eyre::Error};
use std::io::Write;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

mod config;
mod device;

const DEFAULT_PROMPT: &str = "Escribe un haiku sobre programación";
const SEED: u64 = 42;
const MAX_TOKENS: usize = 1000;

struct GenerationConfig {
    name: &'static str,
    temperature: f64,
}

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
    println!("Using model: {}/{}", config.llm.repo, config.llm.file);

    // Download tokenizer and llm from huggingface
    let api = hf_hub::api::tokio::Api::new()?;

    let tokenizer_path = api
        .model(config.tokenizer.repo)
        .get(&config.tokenizer.file)
        .await?;

    let model_path = api
        .repo(hf_hub::Repo::with_revision(
            config.llm.repo,
            hf_hub::RepoType::Model,
            config.llm.branch,
        ))
        .get(&config.llm.file)
        .await?;

    // Detect device for acceleration
    let device = device::load()?;
    if device.is_metal() {
        println!("Using Metal Device");
    } else if device.is_cuda() {
        println!("Using CUDA Device");
    } else {
        println!("Using CPU Device");
    }

    let configs = vec![
        GenerationConfig {
            name: "GREEDY",
            temperature: 1e-10,
        },
        GenerationConfig {
            name: "BALANCED",
            temperature: 0.7,
        },
        GenerationConfig {
            name: "CREATIVE",
            temperature: 1.5,
        },
        GenerationConfig {
            name: "CHAOS",
            temperature: 2.0,
        },
    ];

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
    let vocab = tokenizer.get_vocab(true);
    let eos_token = *vocab.get("<|im_end|>").unwrap();

    let prompt = format!(
        "<|im_start|>user\n{DEFAULT_PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let tokens_encoding = tokenizer.encode(prompt, true).map_err(Error::msg)?;
    let tokens = tokens_encoding.get_ids();

    println!("\nPrompt: {DEFAULT_PROMPT}");

    for gen_config in configs {
        println!(
            "\n\n=== {} (temp {}) ===",
            gen_config.name, gen_config.temperature
        );

        let mut all_tokens = vec![];

        // Read quantized model content
        let mut model_file = std::fs::File::open(&model_path)?;
        let model_content = gguf_file::Content::read(&mut model_file)?;

        let mut model = Qwen3::from_gguf(model_content, &mut model_file, &device)?;

        let mut logits_processor = LogitsProcessor::from_sampling(
            SEED,
            Sampling::All {
                temperature: gen_config.temperature,
            },
        );

        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let output = model.forward(&input, 0)?;
        let output = output.squeeze(0)?;

        let mut next_token = logits_processor.sample(&output)?;
        all_tokens.push(next_token);

        let decoded_token = tokenizer.decode(&[next_token], true).map_err(Error::msg)?;
        print!("{}", decoded_token);
        std::io::stdout().flush()?;

        while all_tokens.len() < MAX_TOKENS && next_token != eos_token {
            let pos = tokens.len() + all_tokens.len();
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let output = model.forward(&input, pos)?;
            let output = output.squeeze(0)?;

            next_token = logits_processor.sample(&output)?;
            all_tokens.push(next_token);

            let decoded_token = tokenizer.decode(&[next_token], true).map_err(Error::msg)?;
            print!("{}", decoded_token);
            std::io::stdout().flush()?;
        }
    }

    Ok(())
}
