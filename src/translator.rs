use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, debug};

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

/// Translate OpenAI embeddings request to Ollama native format
pub fn translate_openai_embeddings_to_ollama(
    openai_req: Value,
    num_ctx: u32,
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

    Ok(OllamaEmbedRequest {
        model: req.model,
        input,
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

        let result = translate_openai_embeddings_to_ollama(openai_req, 8192).unwrap();
        
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

        let result = translate_openai_embeddings_to_ollama(openai_req, 4096).unwrap();
        
        assert_eq!(result.input.len(), 3);
        assert_eq!(result.options.as_ref().unwrap().num_ctx, 4096);
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

