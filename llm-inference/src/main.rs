use std::{io::Write, time::Instant};

use candle_core::{Tensor, quantized::gguf_file};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use color_eyre::{Result, eyre::Error};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;

mod config;
mod device;
mod prompt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    color_eyre::install()?;

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install crypto provider");

    // Cargar configuración
    let config = AppConfig::load()?;

    println!("🔧 Configuración cargada:");
    println!("   Tokenizer: {}", config.tokenizer.repo);
    println!("   Modelo: {}/{}", config.llm.repo, config.llm.file);
    println!("   Temperatura: {}", config.inference.temperature);
    println!("   Top-P: {}", config.inference.top_p);
    println!("   Top-K: {}", config.inference.top_k);
    println!("   Max tokens: {}", config.inference.max_length);

    // Detectar dispositivo
    println!("");
    let device = device::load()?;

    // Descargar tokenizer y modelo desde huggingface
    let api = Api::new()?;

    // Descargar tokenizer
    println!("📥 Descargando tokenizer desde {}", config.tokenizer.repo);
    let tokenizer_path = api
        .model(config.tokenizer.repo)
        .get(&config.tokenizer.file)
        .await?;
    println!("✅ Tokenizer descargado en: {:?}", tokenizer_path);

    // Descargar modelo
    println!(
        "📥 Descargando modelo desde {}/{}",
        config.llm.repo, config.llm.file
    );
    println!("   (Esto puede tardar unos minutos la primera vez)");
    let model_path = api
        .repo(hf_hub::Repo::with_revision(
            config.llm.repo,
            hf_hub::RepoType::Model,
            config.llm.branch,
        ))
        .get(&config.llm.file)
        .await?;

    println!("✅ Modelo descargado en: {:?}", model_path);

    // Cargar tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;

    println!("📚 Tokenizer cargado");
    println!("   Vocabulario: {} tokens", tokenizer.get_vocab_size(true));

    // Obtener el token de fin de secuencia
    let vocab = tokenizer.get_vocab(true);
    let eos_token = *vocab.get(config.tokenizer.eos_token.as_str()).expect(
        format!(
            "El tokenizer debe tener el token {}",
            config.tokenizer.eos_token
        )
        .as_str(),
    );

    println!(
        "   Token EOS: {} (id: {})",
        config.tokenizer.eos_token, eos_token
    );

    // Cargar modelo
    println!("🧠 Cargando modelo (esto puede tardar ~30 segundos)");

    let mut model_file = std::fs::File::open(&model_path)?;
    let model_content = gguf_file::Content::read(&mut model_file)?;
    let mut model = Qwen3::from_gguf(model_content, &mut model_file, &device)?;

    println!("✅ Modelo cargado en memoria");

    // Generar prompt con chat template
    let user_input = "Escribe un haiku sobre programación";
    let prompt = prompt::render(&config.tokenizer.chat_template, user_input)?;

    println!("🎯 Input del usuario:");
    println!("   {}", user_input);
    println!("\n📝 Prompt formateado para el modelo:");
    println!("{}", prompt);

    // Tokenización
    let tokens = tokenizer
        .encode(prompt.clone(), true) // false = no agregar tokens especiales adicionales
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();

    println!("\n🔢 Tokenización:");
    println!("   Total de tokens: {}", tokens.len());
    println!("   Primeros 10 IDs: {:?}", &tokens[..tokens.len().min(10)]);

    // Verificar que el último token sea el que esperamos
    let last_tokens = &tokens[tokens.len().saturating_sub(5)..];
    println!("   Últimos 5 tokens: {:?}", last_tokens);

    // Mostrar token id con su token
    println!("   Token IDs:");
    for token in tokens.iter() {
        println!(
            "   {}: {}",
            token,
            tokenizer.decode(&[*token], false).unwrap()
        );
    }

    // Prefill
    println!("\n🚀 ITERACIÓN 0: PREFILL");
    println!("   Procesando {} tokens del prompt a la vez", tokens.len());

    // Crear tensor con todos los tokens
    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?; // [seq_len] → [1, seq_len] (agregar batch dimension)
    println!("   Shape del input: {:?}", input.shape());

    // Forward pass del modelo
    let start = Instant::now();
    let logits = model.forward(&input, 0)?; // offset = 0 (empezando desde posición 0)
    let prefill_time = start.elapsed();

    println!("   Shape del output: {:?}", logits.shape());
    println!("   ⏱️  Tiempo de prefill: {:.2?}", prefill_time);

    // Crear logits processor con la configuración de sampling
    let mut logits_processor = LogitsProcessor::from_sampling(
        42, // seed (para reproducibilidad)
        Sampling::TopKThenTopP {
            temperature: config.inference.temperature,
            k: config.inference.top_k,
            p: config.inference.top_p,
        },
    );

    println!("\n🎲 SAMPLING");
    println!("   Configuración:");
    println!("     - Temperatura: {}", config.inference.temperature);
    println!("     - Top-k: {}", config.inference.top_k);
    println!("     - Top-p: {}", config.inference.top_p);

    // Squeeze para obtener [vocab_size]
    let logits = logits.squeeze(0)?;

    // Sample el siguiente token
    let mut next_token = logits_processor.sample(&logits)?;

    // Decodificar y mostrar
    let decoded = tokenizer.decode(&[next_token], false).map_err(Error::msg)?;

    println!(
        "   Token seleccionado: \"{}\" (id: {})",
        decoded, next_token
    );
    println!("\n💬 Respuesta del modelo:");
    print!("{}", decoded); // Streaming: mostrar inmediatamente
    std::io::stdout().flush()?;

    // Guardar el token generado
    let mut generated_tokens = vec![next_token];

    // Loop autoregresivo
    println!("\n\n🔄 LOOP AUTOREGRESIVO:");
    println!(
        "   Generando hasta {} tokens o hasta EOS token",
        config.inference.max_length
    );
    println!("   {}", "=".repeat(60));
    print!("{}", decoded);
    std::io::stdout().flush()?;

    while generated_tokens.len() < config.inference.max_length {
        // Calcular la posición actual en la secuencia completa
        let pos = tokens.len() + generated_tokens.len();

        // Crear tensor con UN SOLO TOKEN
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;

        // Forward pass (usa KV cache internamente)
        let logits = model.forward(&input, pos)?;

        // Los logits ya vienen con shape [1, vocab]
        let logits = logits.squeeze(0)?;

        // Sample siguiente token
        next_token = logits_processor.sample(&logits)?;

        // ¿Es el token de fin?
        if next_token == eos_token {
            // Decodificar y mostrar el EOS visualmente
            let decoded = tokenizer.decode(&[next_token], false).map_err(Error::msg)?;
            print!("{}", decoded);
            std::io::stdout().flush()?;

            println!("\n\n   ✅ Generación completada (EOS token detectado)");
            break;
        }

        // Guardar token generado
        generated_tokens.push(next_token);

        // Decodificar y mostrar inmediatamente (streaming)
        let decoded = tokenizer.decode(&[next_token], false).map_err(Error::msg)?;
        print!("{}", decoded);
        std::io::stdout().flush()?;
    }

    // Si terminamos por max_length (timeout)
    if generated_tokens.len() >= config.inference.max_length {
        println!("\n\n   ⚠️  Límite de max_length alcanzado");
    }

    println!("\n{}", "=".repeat(60));
    println!("📊 ESTADÍSTICAS:");
    println!("   Tokens del prompt: {}", tokens.len());
    println!("   Tokens generados: {}", generated_tokens.len());
    println!(
        "   Total procesado: {}",
        tokens.len() + generated_tokens.len()
    );

    Ok(())
}
