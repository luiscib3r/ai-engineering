// src/prompt.rs
use color_eyre::Result;
use tera::{Context, Tera};

pub fn render(template: &str, message: &str) -> Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template("chat", template)?;

    let mut context = Context::new();
    context.insert("message", message);

    Ok(tera.render("chat", &context)?)
}
