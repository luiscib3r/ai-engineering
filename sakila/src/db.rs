use std::path::Path;

use crate::config::DbConfig;
use color_eyre::{Result, eyre::eyre};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sqlx::{Pool, Sqlite, SqlitePool};
use tokio::{fs::File, io::AsyncWriteExt};

pub async fn load(cfg: &DbConfig) -> Result<Pool<Sqlite>> {
    let db_url = format!("sqlite:{}", cfg.file);
    // Check if database file exists
    let file_path = Path::new(&cfg.file);
    // If exists return Ok(())
    if file_path.exists() {
        let pool = SqlitePool::connect(&db_url).await?;
        return Ok(pool);
    }

    if let Some(parent) = file_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tracing::info!("Downloading database from {}...", cfg.url);

    // If file does not exist, download it from cfg.url
    let response = reqwest::get(&cfg.url).await?;
    if !response.status().is_success() {
        return Err(eyre!(
            "Failed to download database: HTTP Status {}",
            response.status()
        ));
    }

    let total_size = response.content_length();

    let pb = if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-")
        );
        pb.set_message("Downloading Sakila DB");
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_message("Downloading Sakila DB (unknown size)");
        pb
    };

    // Save to file
    let mut file = File::create(file_path).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("✓ Database downloaded successfully");

    tracing::info!(
        "Database file downloaded and saved to {} ({} bytes)",
        cfg.file,
        downloaded
    );

    let db_url = format!("sqlite:{}", cfg.file);
    let pool = SqlitePool::connect(&db_url).await?;

    Ok(pool)
}
