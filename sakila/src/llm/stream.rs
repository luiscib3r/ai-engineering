use std::{io::Write, sync::Arc};

use tokenizers::Tokenizer;

pub trait TokenCallback {
    fn on_token(&mut self, token_id: u32);
    fn flush(&mut self);
}

pub struct PrintCallback {
    all_tokens: Vec<u32>,
    tokenizer: Arc<Tokenizer>,
    prev_index: usize,
    current_index: usize,
}

impl PrintCallback {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            all_tokens: Vec::new(),
            tokenizer,
            prev_index: 0,
            current_index: 0,
        }
    }
}

impl TokenCallback for PrintCallback {
    fn on_token(&mut self, token_id: u32) {
        self.all_tokens.push(token_id);

        // Decodificar desde prev_index hasta current_index (texto ya impreso)
        let prev_text = if self.current_index > self.prev_index {
            self.tokenizer
                .decode(&self.all_tokens[self.prev_index..self.current_index], true)
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Decodificar desde prev_index hasta el final (incluye nuevo token)
        let current_text = self
            .tokenizer
            .decode(&self.all_tokens[self.prev_index..], true)
            .unwrap_or_default();

        // Solo imprimir si:
        // 1. El texto creció
        // 2. El último carácter NO es � (replacement char)
        if current_text.len() > prev_text.len() {
            let last_char = current_text.chars().last();

            // Imprimir si el último char NO es � (carácter incompleto)
            if last_char != Some('�') {
                let new_text = &current_text[prev_text.len()..];
                print!("{}", new_text);
                std::io::stdout().flush().ok();

                // Actualizar índices
                self.prev_index = self.current_index;
                self.current_index = self.all_tokens.len();
            }
        }
    }

    fn flush(&mut self) {
        // Decodificar el resto que no se imprimió
        let prev_text = if self.current_index > self.prev_index {
            self.tokenizer
                .decode(&self.all_tokens[self.prev_index..self.current_index], true)
                .unwrap_or_default()
        } else {
            String::new()
        };

        let full_text = self
            .tokenizer
            .decode(&self.all_tokens[self.prev_index..], true)
            .unwrap_or_default();

        if full_text.len() > prev_text.len() {
            let remaining = &full_text[prev_text.len()..];
            print!("{}", remaining);
            std::io::stdout().flush().ok();
        }

        self.all_tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

impl Drop for PrintCallback {
    fn drop(&mut self) {
        self.flush();
    }
}
