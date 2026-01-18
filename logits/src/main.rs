use candle_core::{Tensor, quantized::gguf_file};
use candle_nn::ops::softmax;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use color_eyre::{Result, eyre::Error};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

mod config;
mod device;
mod table;

const DEFAULT_PROMPT: &str = "La capital de Francia es";
const SEED: u64 = 42;

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

    // Read quantized model content
    let mut model_file = std::fs::File::open(&model_path)?;
    let model_content = gguf_file::Content::read(&mut model_file)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
    let mut model = Qwen3::from_gguf(model_content, &mut model_file, &device)?;

    let tokens_encoding = tokenizer
        .encode(DEFAULT_PROMPT.to_string(), true)
        .map_err(Error::msg)?;

    let tokens = tokens_encoding.get_ids();

    println!("\nPrompt: {DEFAULT_PROMPT}");
    println!("Tokens: {:?}", tokens);

    let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let output = model.forward(&input, 0)?;
    let output = output.squeeze(0)?;
    let probabilities = softmax(&output, 0)?;

    let logits = output.to_vec1::<f32>()?;
    let probabilities = probabilities.to_vec1::<f32>()?;

    let mut indexed_logits: Vec<(u32, f32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx as u32, val, probabilities[idx]))
        .collect();

    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_5: Vec<(u32, f32, f32, String)> = indexed_logits
        .iter()
        .take(5)
        .map(|(idx, score, prob)| {
            let idx = *idx;
            let token = tokenizer.decode(&[idx], true).unwrap();
            (idx, *score, *prob, token)
        })
        .collect();

    table::show_logits(top_5);

    // Sampling

    // Greedy sampling (temperature = 0.0)
    let mut logits_processor =
        LogitsProcessor::from_sampling(SEED, Sampling::All { temperature: 1e-10 });

    let next_token_id = logits_processor.sample(&output)?;
    let next_token = tokenizer.decode(&[next_token_id], true).unwrap();

    println!("Greedy sampling: {}", next_token);

    Ok(())
}
