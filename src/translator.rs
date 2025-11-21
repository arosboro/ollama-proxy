use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, debug};
use crate::chunker;

/// OpenAI chat completions request format
#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI chat completions response format
#[derive(Debug, Serialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: OpenAIChatUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatChoice {
    pub index: u32,
    pub message: OpenAIChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Ollama chat request format
#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaChatOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Ollama chat response format
#[derive(Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OpenAIChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// OpenAI embeddings request format
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Serde uses these fields for deserialization
pub struct OpenAIEmbeddingsRequest {
    pub model: String,
    pub input: InputType,
    // Optional fields from OpenAI API spec - kept for proper deserialization
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputType {
    Single(String),
    Multiple(Vec<String>),
}

/// Ollama native embed request format
#[derive(Debug, Serialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OllamaOptions {
    pub num_ctx: u32,
}

/// Ollama native embed response format
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Serde uses these fields for deserialization
pub struct OllamaEmbedResponse {
    // Fields from Ollama API response - kept for proper deserialization
    model: String,
    pub embeddings: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

/// OpenAI embeddings response format
#[derive(Debug, Serialize)]
pub struct OpenAIEmbeddingsResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbedding>,
    pub model: String,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAIEmbedding {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Check if input needs chunking and return chunked inputs
pub fn prepare_embeddings_input(
    input: Vec<String>,
    max_input_length: usize,
    enable_chunking: bool,
) -> Result<Vec<String>, String> {
    // Check for inputs that exceed max length
    let mut needs_chunking = false;
    for (idx, item) in input.iter().enumerate() {
        if item.len() > max_input_length {
            needs_chunking = true;
            info!("   Input {} exceeds max length: {} > {}", idx, item.len(), max_input_length);
        }
    }

    // Apply chunking if needed
    if needs_chunking {
        if !enable_chunking {
            return Err(format!(
                "Input too large ({} characters). Maximum is {} characters. Enable chunking or reduce input size.",
                input.iter().map(|s| s.len()).max().unwrap_or(0),
                max_input_length
            ));
        }

        info!("📦 Chunking large inputs (max length: {})", max_input_length);
        let mut chunked_inputs = Vec::new();
        
        for (idx, item) in input.iter().enumerate() {
            if item.len() > max_input_length {
                let chunks = chunker::chunk_text(item, max_input_length);
                info!("   Input {}: split into {} chunks", idx, chunks.len());
                chunked_inputs.extend(chunks);
            } else {
                chunked_inputs.push(item.clone());
            }
        }
        
        info!("   Total inputs after chunking: {}", chunked_inputs.len());
        Ok(chunked_inputs)
    } else {
        Ok(input)
    }
}

/// Translate OpenAI chat completions request to Ollama native format
pub fn translate_openai_chat_to_ollama(
    openai_req: Value,
    num_ctx: Option<u32>,
) -> Result<OllamaChatRequest, String> {
    let req: OpenAIChatRequest = serde_json::from_value(openai_req)
        .map_err(|e| format!("Failed to parse OpenAI chat request: {}", e))?;

    info!("🔄 Translating OpenAI chat request to Ollama native API");
    info!("   Model: {}", req.model);
    info!("   Messages: {} message(s)", req.messages.len());

    let options = Some(OllamaChatOptions {
        num_ctx,
        num_predict: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
    });

    // Set keep_alive based on context size to prevent model unloading during long requests
    let keep_alive = match num_ctx {
        Some(ctx) if ctx > 32000 => {
            info!("   Setting keep_alive=10m for large context ({})", ctx);
            Some("10m".to_string())
        }
        Some(ctx) if ctx > 16000 => {
            info!("   Setting keep_alive=5m for moderate context ({})", ctx);
            Some("5m".to_string())
        }
        _ => None,
    };

    Ok(OllamaChatRequest {
        model: req.model,
        messages: req.messages,
        stream: req.stream.or(Some(false)),
        options,
        keep_alive,
    })
}

/// Translate Ollama chat response to OpenAI format
pub fn translate_ollama_chat_to_openai(
    ollama_resp: Value,
    _model_fallback: String,
) -> Result<OpenAIChatResponse, String> {
    let resp: OllamaChatResponse = serde_json::from_value(ollama_resp)
        .map_err(|e| format!("Failed to parse Ollama chat response: {}", e))?;

    debug!("🔄 Translating Ollama chat response to OpenAI format");
    debug!("   Model: {}", resp.model);

    // Generate unique ID
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "").chars().take(24).collect::<String>());
    
    // Parse Ollama's ISO8601 timestamp to Unix epoch
    let created = parse_ollama_timestamp(&resp.created_at)
        .unwrap_or_else(|| {
            // Fallback to current time if parsing fails
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

    // Determine finish reason
    let finish_reason = if let Some(reason) = resp.done_reason {
        match reason.as_str() {
            "stop" => "stop".to_string(),
            "length" => "length".to_string(),
            _ => "stop".to_string(),
        }
    } else if resp.done {
        "stop".to_string()
    } else {
        "length".to_string()
    };

    let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
    let completion_tokens = resp.eval_count.unwrap_or(0);

    Ok(OpenAIChatResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: resp.model, // Use the actual model from Ollama response
        choices: vec![OpenAIChatChoice {
            index: 0,
            message: resp.message,
            finish_reason,
        }],
        usage: OpenAIChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

/// Parse Ollama's ISO8601 timestamp to Unix epoch seconds
/// Example: "2025-11-21T16:08:11.735252Z" -> 1763741791
fn parse_ollama_timestamp(timestamp: &str) -> Option<u64> {
    // Simple RFC3339/ISO8601 parsing for Ollama's format
    // Format: YYYY-MM-DDTHH:MM:SS.ffffffZ
    
    // Try parsing as RFC3339 (what Ollama uses)
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(timestamp) {
        return Some(parsed.timestamp() as u64);
    }
    
    // Fallback: try parsing without timezone if it's a simple format
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%S%.fZ") {
        return Some(naive.and_utc().timestamp() as u64);
    }
    
    None
}

/// Translate OpenAI embeddings request to Ollama native format
pub fn translate_openai_embeddings_to_ollama(
    openai_req: Value,
    num_ctx: u32,
    max_input_length: usize,
    enable_chunking: bool,
) -> Result<OllamaEmbedRequest, String> {
    let req: OpenAIEmbeddingsRequest = serde_json::from_value(openai_req)
        .map_err(|e| format!("Failed to parse OpenAI request: {}", e))?;

    // Convert input to vector
    let input = match req.input {
        InputType::Single(s) => vec![s],
        InputType::Multiple(v) => v,
    };

    info!("🔄 Translating OpenAI request to Ollama native API");
    info!("   Model: {}", req.model);
    info!("   Inputs: {} item(s)", input.len());
    info!("   Setting num_ctx: {}", num_ctx);

    // Prepare inputs (with potential chunking)
    let prepared_input = prepare_embeddings_input(input, max_input_length, enable_chunking)?;

    Ok(OllamaEmbedRequest {
        model: req.model,
        input: prepared_input,
        truncate: Some(true),
        options: Some(OllamaOptions { num_ctx }),
        keep_alive: None,
    })
}

/// Translate Ollama native response to OpenAI format
pub fn translate_ollama_embed_to_openai(
    ollama_resp: Value,
    model: String,
) -> Result<OpenAIEmbeddingsResponse, String> {
    let resp: OllamaEmbedResponse = serde_json::from_value(ollama_resp)
        .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

    debug!("🔄 Translating Ollama response to OpenAI format");
    debug!("   Embeddings count: {}", resp.embeddings.len());

    // Convert embeddings to OpenAI format
    let data: Vec<OpenAIEmbedding> = resp
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| OpenAIEmbedding {
            object: "embedding".to_string(),
            embedding,
            index,
        })
        .collect();

    // Calculate usage (approximate)
    let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);

    Ok(OpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data,
        model,
        usage: OpenAIUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
}

/// Determine if translation is needed based on the endpoint
pub fn needs_translation(path: &str) -> bool {
    matches!(path, "/v1/embeddings" | "/v1/chat/completions")
}

/// Get the corresponding Ollama native endpoint for an OpenAI endpoint
pub fn get_ollama_endpoint(openai_path: &str) -> &str {
    match openai_path {
        "/v1/embeddings" => "/api/embed",
        "/v1/chat/completions" => "/api/chat",
        _ => openai_path, // Pass through for endpoints that don't need translation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_translate_openai_single_input() {
        let openai_req = json!({
            "model": "nomic-embed-text",
            "input": "Hello world"
        });

        let result = translate_openai_embeddings_to_ollama(openai_req, 8192, 2000, true).unwrap();
        
        assert_eq!(result.model, "nomic-embed-text");
        assert_eq!(result.input.len(), 1);
        assert_eq!(result.input[0], "Hello world");
        assert_eq!(result.options.as_ref().unwrap().num_ctx, 8192);
        assert_eq!(result.truncate, Some(true));
    }

    #[test]
    fn test_translate_openai_multiple_inputs() {
        let openai_req = json!({
            "model": "nomic-embed-text",
            "input": ["Hello", "World", "Test"]
        });

        let result = translate_openai_embeddings_to_ollama(openai_req, 4096, 2000, true).unwrap();
        
        assert_eq!(result.input.len(), 3);
        assert_eq!(result.options.as_ref().unwrap().num_ctx, 4096);
    }

    #[test]
    fn test_translate_with_chunking() {
        let long_text = "a".repeat(5000);
        let openai_req = json!({
            "model": "nomic-embed-text",
            "input": long_text
        });

        let result = translate_openai_embeddings_to_ollama(openai_req, 8192, 2000, true).unwrap();
        
        // Should be split into multiple chunks
        assert!(result.input.len() > 1);
        
        // Each chunk should not exceed max length
        for chunk in &result.input {
            assert!(chunk.len() <= 2000);
        }
    }

    #[test]
    fn test_translate_chunking_disabled_error() {
        let long_text = "a".repeat(5000);
        let openai_req = json!({
            "model": "nomic-embed-text",
            "input": long_text
        });

        let result = translate_openai_embeddings_to_ollama(openai_req, 8192, 2000, false);
        
        // Should return error when chunking is disabled
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_translate_ollama_response() {
        let ollama_resp = json!({
            "model": "nomic-embed-text",
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ],
            "prompt_eval_count": 10
        });

        let result = translate_ollama_embed_to_openai(
            ollama_resp,
            "nomic-embed-text".to_string()
        ).unwrap();

        assert_eq!(result.object, "list");
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.data[0].index, 0);
        assert_eq!(result.data[1].index, 1);
        assert_eq!(result.usage.prompt_tokens, 10);
    }

    #[test]
    fn test_needs_translation() {
        assert!(needs_translation("/v1/embeddings"));
        assert!(needs_translation("/v1/chat/completions"));
        assert!(!needs_translation("/v1/models"));
        assert!(!needs_translation("/api/embed"));
    }

    #[test]
    fn test_get_ollama_endpoint() {
        assert_eq!(get_ollama_endpoint("/v1/embeddings"), "/api/embed");
        assert_eq!(get_ollama_endpoint("/v1/chat/completions"), "/api/chat");
        assert_eq!(get_ollama_endpoint("/v1/models"), "/v1/models"); // Passthrough
    }
}

