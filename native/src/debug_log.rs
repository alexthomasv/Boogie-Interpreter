use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn enabled(category: &str) -> Option<String> {
    let path = env::var("SWOOSH_DEBUG_LOG").ok()?;
    let cats = env::var("SWOOSH_DEBUG_CATEGORIES").unwrap_or_else(|_| "all".to_string());
    if cats.split(',').any(|c| {
        let c = c.trim();
        c == "all" || c == category
    }) {
        Some(path)
    } else {
        None
    }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub fn event(category: &str, event: &str, fields: &[(&str, String)]) {
    let Some(path) = enabled(category) else {
        return;
    };
    let ts_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let run_id = env::var("SWOOSH_DEBUG_RUN_ID").unwrap_or_else(|_| "native".to_string());
    let mut line = String::new();
    line.push('{');
    line.push_str(&format!("\"category\":\"{}\"", escape_json(category)));
    line.push_str(&format!(",\"event\":\"{}\"", escape_json(event)));
    line.push_str(&format!(",\"pid\":{}", std::process::id()));
    line.push_str(&format!(",\"run_id\":\"{}\"", escape_json(&run_id)));
    line.push_str(&format!(",\"ts_ns\":{}", ts_ns));
    for (key, value) in fields {
        line.push_str(&format!(
            ",\"{}\":\"{}\"",
            escape_json(key),
            escape_json(value)
        ));
    }
    line.push_str("}\n");

    if let Ok(mut fh) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = fh.write_all(line.as_bytes());
    }
}

