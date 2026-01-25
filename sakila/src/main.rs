use crate::{agent::Agent, config::AppConfig};
use color_eyre::Result;
use tracing_subscriber::EnvFilter;

mod agent;
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
    let db = db::load(&config.db).await?;

    // Load llm
    let llm = llm::Llm::load(&config).await?;

    // Build agent
    let mut agent = Agent::new(&config.agent, llm, db);

    println!("Sakila Chat (type /exit to quit)\n");

    agent.run().await?;

    Ok(())
}
