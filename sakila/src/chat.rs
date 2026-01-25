#[derive(Debug, Clone)]
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
}
