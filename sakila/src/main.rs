use std::io::{self, Write};

use color_eyre::Result;
use colored::Colorize;
use tracing_subscriber::EnvFilter;

use crate::{config::AppConfig, llm::stream::PrintCallback};

mod chat;
mod config;
mod db;
mod device;
mod llm;

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

    // Load database
    let _ = db::load(&config.db).await?;

    // Load llm
    let mut llm = llm::Llm::load(&config).await?;

    println!("Sakila Chat (type /exit to quit)\n");

    // Chat
    loop {
        println!("\n─ {} ─", "You".bright_cyan().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            break;
        }

        let user_message = chat::Message::User {
            content: input.to_string(),
        };

        println!("\n─ {} ─", "Assistant".bright_cyan().bold());
        io::stdout().flush()?;

        let mut callback = PrintCallback::new(llm.get_tokenizer());
        llm.chat_stream(&vec![user_message], &mut callback)?;

        println!();
    }

    Ok(())
}
