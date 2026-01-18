use tabled::{Table, Tabled};

#[derive(Tabled)]
struct TopLogit {
    #[tabled(rename = "Rank")]
    rank: usize,
    #[tabled(rename = "Index")]
    index: u32,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Probability")]
    probability: String,
    #[tabled(rename = "Token")]
    token: String,
}

pub fn show_logits(top_5: Vec<(u32, f32, f32, String)>) {
    let table_data: Vec<TopLogit> = top_5
        .iter()
        .enumerate()
        .map(|(rank, (idx, score, prob, token))| TopLogit {
            rank: rank + 1,
            index: *idx,
            score: format!("{:.4}", score),
            probability: format!("{:.2}", prob),
            token: token.to_string(),
        })
        .collect();

    let table = Table::new(table_data);
    println!("\n{}", table);
}
