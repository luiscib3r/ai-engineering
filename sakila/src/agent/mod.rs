use color_eyre::Result;
use sqlx::{Pool, Sqlite};
use std::io::{Write, stdin, stdout};

use colored::Colorize;

use crate::{
    chat::Message,
    config::AgentConfig,
    llm::{Llm, stream::PrintCallback},
};

pub struct Agent {
    llm: Llm,
    db: Pool<Sqlite>,
    memory: Vec<Message>,
    max_iterations: usize,
    last_sql: Option<String>,
}

mod sql;

impl Agent {
    pub fn new(cfg: &AgentConfig, llm: Llm, db: Pool<Sqlite>) -> Self {
        let system_message = Message::System {
            content: cfg.system_prompt.clone(),
        };
        Self {
            llm,
            db,
            memory: vec![system_message],
            max_iterations: cfg.max_iterations,
            last_sql: None,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        println!("{}", "🎬 Sakila Agent".bright_magenta().bold());
        println!("{}", "Type /exit to quit\n".dimmed());

        // Chat
        loop {
            // User input
            println!("{}", format!("\n─ {} ─", "You").bright_cyan().bold());
            stdout().flush()?;

            let mut input = String::new();
            stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            if input == "/exit" {
                println!("{}", "👋 Adiós!".bright_yellow());
                break;
            }

            // Add user message to memory
            let user_message = Message::User {
                content: input.to_string(),
            };
            self.memory.push(user_message);

            let mut iterations = 0;
            while iterations < self.max_iterations && self.run_agent().await? {
                iterations += 1;
            }

            if iterations >= self.max_iterations {
                println!(
                    "{}",
                    "\n⚠️  Límite de iteraciones alcanzado".bright_yellow()
                );
            }

            // Clean system tools messages
            self.memory.retain(|msg| match msg {
                Message::System { content } => {
                    !content.starts_with("<sql_result>") && !content.starts_with("<sql_error>")
                }
                _ => true,
            });
            self.last_sql = None;

            println!();
        }

        Ok(())
    }

    async fn run_agent(&mut self) -> Result<bool> {
        // Generate response
        println!("{}", format!("\n─ {} ─", "Assistant").bright_cyan().bold());
        stdout().flush()?;

        let mut callback = PrintCallback::new(self.llm.get_tokenizer());
        let assistant_message = self.llm.chat_stream(&self.memory, &mut callback)?;
        self.memory.push(assistant_message.clone());

        let content = match assistant_message {
            Message::Assistant { content } => content,
            _ => "".into(),
        };

        if let Some(sql) = sql::extract_sql(&content) {
            println!("\n");
            println!("{}", "🔍 Ejecutando SQL...".bright_yellow());
            println!("{}", format!("   {}", sql.dimmed()));

            if self.last_sql.as_ref() == Some(&sql) {
                println!(
                    "{}",
                    "\n⚠️  El modelo está generando la misma query. Deteniendo ejecución de sql."
                        .bright_yellow()
                );
                // Agregar mensaje de ayuda al contexto
                let hint_message = Message::System {
                    content: format!(
                        "<sql_result>
                        Ya ejecutaste esta query: {}\n\
                        El resultado no fue el esperado. Intenta:\n\
                        1. Una query DIFERENTE\n\
                        2. Simplificar la query\n\
                        3. Explicar al usuario que hay un problema con los datos
                        </sql_result>",
                        sql
                    ),
                };
                self.memory.push(hint_message);
                return Ok(true);
            }

            self.last_sql = Some(sql.clone());

            // Execute SQL
            self.run_sql(&sql).await;

            return Ok(true);
        };

        Ok(false)
    }

    async fn run_sql(&mut self, sql: &str) {
        match sql::run_query(&self.db, &sql).await {
            Ok(results) => {
                let formatted = sql::format_results(&results);
                println!("{}", "✅ Query ejecutada".bright_green());
                println!("{}", formatted.dimmed());
                // Add results to memory for model context
                let result_message = Message::System {
                    content: format!("<sql_result>\n{}\n</sql_result>", formatted),
                };

                self.memory.push(result_message);
            }
            Err(err) => {
                println!("{}", format!("❌ Error: {}", err).bright_red());

                // Add error to memory
                let error_message = Message::System {
                    content: format!("<sql_error>{}</sql_error>", err),
                };
                self.memory.push(error_message);
            }
        };
    }
}
