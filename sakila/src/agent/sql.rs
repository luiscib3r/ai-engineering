use color_eyre::{Result, eyre::eyre};
use regex::Regex;
use serde_json::{Value, json};
use sqlx::{Column, Pool, Row, Sqlite};

pub(super) fn extract_sql(text: &str) -> Option<String> {
    let re = Regex::new(r"(?s)<sql>(.*?)</sql>").ok()?;
    re.captures(text)?
        .get(1)
        .map(|m| m.as_str().trim().to_string())
}

pub(super) async fn run_query(pool: &Pool<Sqlite>, query: &str) -> Result<Value> {
    let query_upper = query.trim().to_uppercase();
    if query_upper.starts_with("DROP")
        || query_upper.starts_with("DELETE")
        || query_upper.starts_with("UPDATE")
        || query_upper.starts_with("INSERT")
    {
        return Err(eyre!("Query no permitida: solo SELECT"));
    }

    // Ejecutar
    let rows = sqlx::query(query).fetch_all(pool).await?;

    if rows.is_empty() {
        return Ok(json!({"rows": [], "count": 0}));
    }

    let mut results = Vec::new();
    for row in &rows {
        let mut obj = serde_json::Map::new();

        for (i, column) in row.columns().iter().enumerate() {
            let value = if let Ok(v) = row.try_get::<i64, _>(i) {
                json!(v)
            } else if let Ok(v) = row.try_get::<f64, _>(i) {
                json!(v)
            } else if let Ok(v) = row.try_get::<String, _>(i) {
                json!(v)
            } else if let Ok(v) = row.try_get::<bool, _>(i) {
                json!(v)
            } else {
                json!(null)
            };

            obj.insert(column.name().to_string(), value);
        }

        results.push(Value::Object(obj));
    }

    Ok(json!({
        "rows": results,
        "count": results.len()
    }))
}

pub(super) fn format_results(result: &Value) -> String {
    let count = result["count"].as_u64().unwrap_or(0);

    if count == 0 {
        return "No se encontraron resultados.".to_string();
    }

    let rows = result["rows"].as_array().unwrap();

    // Si es una sola fila con un solo valor, output simple
    if count == 1 {
        let row = &rows[0];
        if let Some(obj) = row.as_object() {
            if obj.len() == 1 {
                let value = obj.values().next().unwrap();
                return format!("Resultado: {}", value);
            }
        }
    }

    // Sino, formato tabla
    let mut output = format!("Resultados ({} filas):\n", count);

    for (i, row) in rows.iter().take(10).enumerate() {
        output.push_str(&format!("{}. ", i + 1));
        if let Some(obj) = row.as_object() {
            let values: Vec<String> = obj
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v.as_str().unwrap_or("NULL")))
                .collect();
            output.push_str(&values.join(", "));
        }
        output.push('\n');
    }

    if count > 10 {
        output.push_str(&format!("... y {} filas más\n", count - 10));
    }

    output
}
