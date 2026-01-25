use std::sync::Arc;

use crate::{chat::Message, config::AppConfig, device};
use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use color_eyre::{Result, eyre::Error};
use hf_hub::api::tokio::Api;
use tera::{Context, Tera};
use tokenizers::Tokenizer;

pub mod stream;

pub struct Llm {
    device: Device,
    tokenizer: Arc<Tokenizer>,
    eos_token: u32,
    model: Qwen3,
    logits_processor: LogitsProcessor,
    prompt: Tera,
    start_completion: String,
    max_length: usize,
    kv_cache_offset: usize,
    banned_tokens: Vec<u32>,
}

impl Llm {
    pub async fn load(config: &AppConfig) -> Result<Self> {
        // Setup device
        let device = device::load()?;

        // Load models and tokenizer from hf
        let api = Api::new()?;

        tracing::info!("📥 Descargando tokenizer desde {}", config.tokenizer.repo);
        let tokenizer_path = api
            .model(config.tokenizer.repo.clone())
            .get(&config.tokenizer.file)
            .await?;
        tracing::info!("✅ Tokenizer descargado en: {:?}", tokenizer_path);

        tracing::info!(
            "📥 Descargando modelo desde {}/{}",
            config.llm.repo,
            config.llm.file
        );
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                config.llm.repo.clone(),
                hf_hub::RepoType::Model,
                config.llm.branch.clone(),
            ))
            .get(&config.llm.file)
            .await?;
        tracing::info!("✅ Modelo descargado en: {:?}", model_path);

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
        let tokenizer = Arc::new(tokenizer);
        let vocab = tokenizer.get_vocab(true);
        let eos_token = *vocab.get(config.tokenizer.eos_token.as_str()).expect(
            format!(
                "El tokenizer debe tener el token {}",
                config.tokenizer.eos_token
            )
            .as_str(),
        );

        let banned_tokens: Vec<u32> = config
            .tokenizer
            .banned_tokens
            .iter()
            .filter_map(|token| {
                let id = vocab.get(token.as_str()).copied();
                if id.is_none() {
                    tracing::warn!("⚠️  Token baneado '{}' no existe en vocabulario", token);
                }
                id
            })
            .collect();

        let mut model_file = std::fs::File::open(&model_path)?;
        let model_content = gguf_file::Content::read(&mut model_file)?;
        let model = Qwen3::from_gguf(model_content, &mut model_file, &device)?;

        let logits_processor = LogitsProcessor::from_sampling(
            config.llm.seed,
            Sampling::TopKThenTopP {
                temperature: config.inference.temperature,
                k: config.inference.top_k,
                p: config.inference.top_p,
            },
        );

        let mut prompt = Tera::default();
        prompt.add_raw_template("user", &config.tokenizer.user_template)?;
        prompt.add_raw_template("assistant", &config.tokenizer.assistant_template)?;
        prompt.add_raw_template("system", &config.tokenizer.system_template)?;

        Ok(Self {
            device,
            tokenizer,
            eos_token,
            model,
            logits_processor,
            prompt,
            start_completion: config.tokenizer.start_completion.clone(),
            max_length: config.inference.max_length.clone(),
            kv_cache_offset: 0,
            banned_tokens,
        })
    }

    pub fn get_tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    #[allow(dead_code)]
    pub fn chat(&mut self, messages: &Vec<Message>) -> Result<Message> {
        let input_text = self.render(messages)?;
        let content = self.run(&input_text, None)?;

        Ok(Message::Assistant { content })
    }

    pub fn chat_stream(
        &mut self,
        messages: &Vec<Message>,
        callback: &mut dyn stream::TokenCallback,
    ) -> Result<Message> {
        let input_text = self.render(messages)?;
        let content = self.run(&input_text, Some(callback))?;
        Ok(Message::Assistant { content })
    }

    fn run(
        &mut self,
        text: &str,
        mut callback: Option<&mut dyn stream::TokenCallback>,
    ) -> Result<String> {
        let mut next_token = self.prefill(text)?;
        let mut generated_tokens = vec![next_token];

        if let Some(cb) = callback.as_mut() {
            cb.on_token(next_token);
        }

        while generated_tokens.len() < self.max_length {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, self.kv_cache_offset)?;
            let logits = logits.squeeze(0)?;
            let logits = self.apply_logits_bias(&logits)?;
            next_token = self.logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
            self.kv_cache_offset += 1;

            if let Some(cb) = callback.as_mut() {
                cb.on_token(next_token);
            }

            if next_token == self.eos_token {
                break;
            }
        }

        if let Some(cb) = callback.as_mut() {
            cb.flush();
        }

        let result = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(Error::msg)?;

        Ok(result)
    }

    fn prefill(&mut self, text: &str) -> Result<u32> {
        let encoded = self.tokenizer.encode(text, true).map_err(Error::msg)?;
        let tokens = encoded.get_ids();
        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.kv_cache_offset)?;
        let logits = logits.squeeze(0)?;
        let logits = self.apply_logits_bias(&logits)?;

        // Recalculate the KV cache size
        let next_token = self.logits_processor.sample(&logits)?;
        self.kv_cache_offset += tokens.len();
        Ok(next_token)
    }

    fn apply_logits_bias(&self, logits: &Tensor) -> Result<Tensor> {
        let mut logits_vec = logits.to_vec1::<f32>()?;

        for &token_id in &self.banned_tokens {
            logits_vec[token_id as usize] = f32::NEG_INFINITY;
        }

        Ok(Tensor::from_vec(logits_vec, logits.shape(), &self.device)?)
    }

    fn render(&self, messages: &Vec<Message>) -> Result<String> {
        let text_messages: Vec<String> = messages
            .iter()
            .map(|message| match message {
                Message::System { content } => self.render_system(content),
                Message::User { content } => self.render_user(content),
                Message::Assistant { content } => self.render_assistant(content),
            })
            .collect::<Result<Vec<String>>>()?;

        let mut rendered = text_messages.join("");
        rendered.push_str(&self.start_completion);
        Ok(rendered)
    }

    fn render_system(&self, message: &str) -> Result<String> {
        let mut context = Context::new();
        context.insert("message", message);
        let msg = self.prompt.render("system", &context)?;
        Ok(msg)
    }

    fn render_assistant(&self, message: &str) -> Result<String> {
        let mut context = Context::new();
        context.insert("message", message);
        let msg = self.prompt.render("assistant", &context)?;
        Ok(msg)
    }

    fn render_user(&self, message: &str) -> Result<String> {
        let mut context = Context::new();
        context.insert("message", message);
        let msg = self.prompt.render("user", &context)?;
        Ok(msg)
    }
}
