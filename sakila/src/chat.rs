#[derive(Debug)]
pub enum Message {
    User { content: String },
    Assistant { content: String },
}
