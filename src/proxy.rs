use axum::{
    extract::State,
    http::{Request, Response, StatusCode},
    body::Body,
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tracing::{info, warn, error, debug};
use serde_json::Value;

use crate::model_metadata::ModelMetadataCache;
use crate::modifier::apply_modifiers;
use crate::translator::{
    needs_translation, get_ollama_endpoint,
    translate_openai_embeddings_to_ollama, translate_ollama_embed_to_openai,
    translate_openai_chat_to_ollama, translate_ollama_chat_to_openai,
    OllamaEmbedRequest, OllamaOptions, prepare_embeddings_input, InputType,
};

#[derive(Clone)]
pub struct ProxyState {
    pub ollama_host: String,
    pub client: reqwest::Client,
    pub metadata_cache: Arc<ModelMetadataCache>,
    pub max_embedding_input_length: usize,
    pub enable_auto_chunking: bool,
    pub max_context_override: u32,
    pub request_timeout_seconds: u64,
}

impl ProxyState {
    pub fn new(
        ollama_host: String,
        max_embedding_input_length: usize,
        enable_auto_chunking: bool,
        max_context_override: u32,
        request_timeout_seconds: u64,
    ) -> Self {
        Self {
            ollama_host: ollama_host.clone(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(request_timeout_seconds))
                .build()
                .expect("Failed to build HTTP client"),
            metadata_cache: Arc::new(ModelMetadataCache::new(ollama_host)),
            max_embedding_input_length,
            enable_auto_chunking,
            max_context_override,
            request_timeout_seconds,
        }
    }
}

pub async fn proxy_handler(
    State(state): State<ProxyState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();
    let query = uri.query().unwrap_or("");
    
    info!("📨 Incoming request: {} {}{}", 
        method, 
        path,
        if query.is_empty() { String::new() } else { format!("?{}", query) }
    );

    // Collect headers
    let headers = req.headers().clone();
    debug!("Headers: {:?}", headers);

    // Read the body
    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Check if this is an OpenAI endpoint that needs translation
    if needs_translation(&path) {
        return handle_translated_request(state, &path, body_bytes, headers).await;
    }

    // For non-translated requests, use the original logic
    handle_standard_request(state, &path, query, method, body_bytes, headers).await
}

/// Handle requests that need OpenAI to Ollama translation
async fn handle_translated_request(
    state: ProxyState,
    path: &str,
    body_bytes: bytes::Bytes,
    _headers: axum::http::HeaderMap,
) -> Result<Response<Body>, StatusCode> {
    // Parse the incoming OpenAI request
    let body_json: Value = match serde_json::from_slice(&body_bytes) {
        Ok(json) => {
            info!("📋 OpenAI Request body: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
            json
        }
        Err(e) => {
            error!("Failed to parse OpenAI request body: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Extract model name
    let model_name = match extract_model_name(&body_json) {
        Some(name) => name,
        None => {
            error!("No model specified in request");
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    info!("🔍 Detected model: {}", model_name);

    // Fetch model metadata to get proper context length
    let metadata = match state.metadata_cache.get_model_info(&model_name).await {
        Ok(meta) => {
            info!("📊 Model metadata - n_ctx_train: {}", meta.n_ctx_train);
            meta
        }
        Err(e) => {
            warn!("⚠️  Could not fetch model metadata: {}, using default", e);
            crate::model_metadata::ModelMetadata::default()
        }
    };

    // Handle embeddings specially with chunking support
    if path == "/v1/embeddings" {
        return handle_embeddings_with_chunking(state, body_json, metadata.n_ctx_train, model_name).await;
    }

    // Handle chat completions
    if path == "/v1/chat/completions" {
        // Calculate effective context: respect user's MAX_CONTEXT_OVERRIDE
        let effective_ctx = metadata.n_ctx_train.min(state.max_context_override);
        info!("🎯 Context calculation: model={}, override={}, effective={}", 
            metadata.n_ctx_train, state.max_context_override, effective_ctx);
        
        return handle_chat_completions(state, body_json, Some(effective_ctx), model_name, metadata).await;
    }

    error!("Translation not implemented for path: {}", path);
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Handle embeddings requests with automatic chunking for large inputs
async fn handle_embeddings_with_chunking(
    state: ProxyState,
    body_json: Value,
    num_ctx: u32,
    model_name: String,
) -> Result<Response<Body>, StatusCode> {
    // Parse input
    #[derive(serde::Deserialize)]
    struct EmbedReq {
        input: InputType,
    }
    
    let req: EmbedReq = match serde_json::from_value(body_json.clone()) {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to parse embeddings request: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Convert input to vector
    let inputs = match req.input {
        InputType::Single(s) => vec![s],
        InputType::Multiple(v) => v,
    };

    // Check if chunking is needed
    let max_len = state.max_embedding_input_length;
    let needs_chunking = inputs.iter().any(|s| s.len() > max_len);

    if !needs_chunking {
        // No chunking needed, process normally
        return handle_single_embeddings_request(state, body_json, num_ctx, model_name).await;
    }

    // Chunking needed - process each chunk separately
    info!("🔀 Processing large input with sequential chunking");
    
    // Prepare chunked inputs
    let chunked_inputs = match prepare_embeddings_input(
        inputs,
        max_len,
        state.enable_auto_chunking,
    ) {
        Ok(chunks) => chunks,
        Err(e) => {
            error!("Chunking failed: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    info!("📦 Processing {} chunks sequentially", chunked_inputs.len());

    // Process each chunk as a separate request
    let mut all_embeddings = Vec::new();
    let target_path = get_ollama_endpoint("/v1/embeddings");
    let target_url = format!("{}{}", state.ollama_host, target_path);

    for (idx, chunk) in chunked_inputs.iter().enumerate() {
        info!("   Processing chunk {}/{}", idx + 1, chunked_inputs.len());
        
        let ollama_req = OllamaEmbedRequest {
            model: model_name.clone(),
            input: vec![chunk.clone()],
            truncate: Some(true),
            options: Some(OllamaOptions { num_ctx }),
            keep_alive: None,
        };

        let req_body = match serde_json::to_vec(&ollama_req) {
            Ok(b) => b,
            Err(e) => {
                error!("Failed to serialize chunk request: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        };

        // Send request with retry
        let response = match send_with_retry(&state.client, &target_url, req_body, 2).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to process chunk {}: {}", idx + 1, e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        };

        let status = response.status();
        if !status.is_success() {
            if status == StatusCode::INTERNAL_SERVER_ERROR {
                error!("❌ Ollama server error (500) for chunk {}: This may indicate memory allocation failure", idx + 1);
                error!("   Try reducing MAX_EMBEDDING_INPUT_LENGTH or check Ollama logs");
            } else {
                error!("Ollama returned error for chunk {}: {}", idx + 1, status);
            }
            let error_body = response.bytes().await.unwrap_or_default();
            let error_text = String::from_utf8_lossy(&error_body);
            if !error_text.is_empty() {
                error!("   Error details: {}", error_text);
            }
            return Ok(Response::builder()
                .status(status)
                .header("Content-Type", "application/json")
                .body(Body::from(error_body))
                .unwrap());
        }

        // Parse response
        let response_bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Failed to read chunk {} response: {}", idx + 1, e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        };

        let ollama_resp: Value = match serde_json::from_slice(&response_bytes) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to parse chunk {} response: {}", idx + 1, e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        };

        // Extract embeddings
        if let Some(embeddings) = ollama_resp.get("embeddings").and_then(|e| e.as_array()) {
            for embedding in embeddings {
                if let Some(vec) = embedding.as_array() {
                    let float_vec: Vec<f32> = vec.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    all_embeddings.push(float_vec);
                }
            }
        }
    }

    info!("✅ Collected {} embeddings from chunks", all_embeddings.len());

    // Combine embeddings by averaging
    let combined_embedding = if all_embeddings.is_empty() {
        vec![]
    } else {
        let dim = all_embeddings[0].len();
        let mut combined = vec![0.0f32; dim];
        
        for embedding in &all_embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                if i < dim {
                    combined[i] += val;
                }
            }
        }
        
        // Average
        for val in &mut combined {
            *val /= all_embeddings.len() as f32;
        }
        
        combined
    };

    // Build OpenAI response
    let openai_resp = crate::translator::OpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: vec![crate::translator::OpenAIEmbedding {
            object: "embedding".to_string(),
            embedding: combined_embedding,
            index: 0,
        }],
        model: model_name,
        usage: crate::translator::OpenAIUsage {
            prompt_tokens: all_embeddings.len() as u32 * 10, // Approximate
            total_tokens: all_embeddings.len() as u32 * 10,
        },
    };

    let response_body = match serde_json::to_vec(&openai_resp) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize response: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(response_body))
        .unwrap())
}

/// Handle single (non-chunked) embeddings request
async fn handle_single_embeddings_request(
    state: ProxyState,
    body_json: Value,
    num_ctx: u32,
    model_name: String,
) -> Result<Response<Body>, StatusCode> {
    let ollama_req = match translate_openai_embeddings_to_ollama(
        body_json,
        num_ctx,
        state.max_embedding_input_length,
        state.enable_auto_chunking,
    ) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to translate request: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    let body = match serde_json::to_vec(&ollama_req) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize request: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    info!("📤 Translated request: {}", serde_json::to_string_pretty(&ollama_req).unwrap_or_default());

    let target_path = get_ollama_endpoint("/v1/embeddings");
    let target_url = format!("{}{}", state.ollama_host, target_path);
    info!("🔄 Forwarding to Ollama native API: {}", target_url);

    let response = match state.client.post(&target_url)
        .body(body)
        .header("Content-Type", "application/json")
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            error!("❌ Failed to proxy request: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = response.status();
    info!("📬 Ollama response status: {}", status);

    if !status.is_success() {
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            error!("❌ Ollama server error (500): This may indicate:");
            error!("   - Input too large (try enabling chunking or reducing input)");
            error!("   - Model memory allocation failure");
            error!("   - Check Ollama logs for details: ~/.ollama/logs/server.log");
        } else {
            error!("Ollama returned error status: {}", status);
        }
        let error_body = response.bytes().await.unwrap_or_default();
        let error_text = String::from_utf8_lossy(&error_body);
        if !error_text.is_empty() {
            debug!("   Error details: {}", error_text);
        }
        return Ok(Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(Body::from(error_body))
            .unwrap());
    }

    let response_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read response body: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let ollama_resp: Value = match serde_json::from_slice(&response_bytes) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to parse Ollama response: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    debug!("📥 Ollama response: {}", serde_json::to_string_pretty(&ollama_resp).unwrap_or_default());

    let openai_resp = match translate_ollama_embed_to_openai(ollama_resp, model_name) {
        Ok(resp) => resp,
        Err(e) => {
            error!("Failed to translate response: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    info!("✅ Translated response back to OpenAI format");

    let response_body = match serde_json::to_vec(&openai_resp) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize OpenAI response: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(response_body))
        .unwrap())
}

/// Handle chat completions request
async fn handle_chat_completions(
    state: ProxyState,
    body_json: Value,
    num_ctx: Option<u32>,
    model_name: String,
    metadata: crate::model_metadata::ModelMetadata,
) -> Result<Response<Body>, StatusCode> {
    // Check if streaming is requested
    if let Some(stream) = body_json.get("stream").and_then(|s| s.as_bool()) {
        if stream {
            warn!("⚠️  Streaming with OpenAI→Ollama translation is not yet supported");
            warn!("   Recommendation: Use /api/chat endpoint directly for streaming, or set stream=false");
            warn!("   Falling back to non-streaming mode");
        }
    }
    
    let ollama_req = match translate_openai_chat_to_ollama(body_json, num_ctx) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to translate chat request: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Convert to Value for modifier application
    let mut ollama_req_json = match serde_json::to_value(&ollama_req) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to convert chat request to JSON: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Apply modifiers (context limits, num_predict, etc.)
    info!("🔧 Applying modifiers to translated chat request");
    let modified = apply_modifiers(&mut ollama_req_json, &metadata, state.max_context_override);
    if modified {
        info!("✏️  Request modified by modifiers");
    }

    let body = match serde_json::to_vec(&ollama_req_json) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize chat request: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    info!("📤 Final chat request: {}", serde_json::to_string_pretty(&ollama_req_json).unwrap_or_default());

    let target_path = get_ollama_endpoint("/v1/chat/completions");
    let target_url = format!("{}{}", state.ollama_host, target_path);
    info!("🔄 Forwarding to Ollama native API: {}", target_url);

    let response = match state.client.post(&target_url)
        .body(body)
        .header("Content-Type", "application/json")
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            error!("❌ Failed to proxy chat request: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = response.status();
    info!("📬 Ollama chat response status: {}", status);

    if !status.is_success() {
        error!("Ollama returned error status: {}", status);
        let error_body = response.bytes().await.unwrap_or_default();
        let error_text = String::from_utf8_lossy(&error_body);
        if !error_text.is_empty() {
            debug!("   Error details: {}", error_text);
        }
        return Ok(Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(Body::from(error_body))
            .unwrap());
    }

    let response_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read chat response body: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let ollama_resp: Value = match serde_json::from_slice(&response_bytes) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to parse Ollama chat response: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    debug!("📥 Ollama chat response: {}", serde_json::to_string_pretty(&ollama_resp).unwrap_or_default());

    let openai_resp = match translate_ollama_chat_to_openai(ollama_resp, model_name) {
        Ok(resp) => resp,
        Err(e) => {
            error!("Failed to translate chat response: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    info!("✅ Translated chat response back to OpenAI format");

    let response_body = match serde_json::to_vec(&openai_resp) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize OpenAI chat response: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(response_body))
        .unwrap())
}

/// Send request with retry logic
async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    body: Vec<u8>,
    max_retries: usize,
) -> Result<reqwest::Response, String> {
    let mut attempts = 0;
    
    loop {
        attempts += 1;
        
        match client.post(url)
            .body(body.clone())
            .header("Content-Type", "application/json")
            .send()
            .await
        {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                if e.is_timeout() {
                    return Err(format!("Request timed out: {}", e));
                }
                if attempts >= max_retries {
                    return Err(format!("Failed after {} attempts: {}", attempts, e));
                }
                warn!("Request failed (attempt {}), retrying: {}", attempts, e);
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        }
    }
}

/// Handle standard requests (no translation needed)
async fn handle_standard_request(
    state: ProxyState,
    path: &str,
    query: &str,
    method: axum::http::Method,
    body_bytes: bytes::Bytes,
    headers: axum::http::HeaderMap,
) -> Result<Response<Body>, StatusCode> {
    // Try to parse as JSON for logging and modification
    let mut body_json: Option<Value> = if !body_bytes.is_empty() {
        match serde_json::from_slice(&body_bytes) {
            Ok(json) => {
                info!("📋 Request body: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
                Some(json)
            }
            Err(_) => {
                debug!("Body is not JSON or empty");
                None
            }
        }
    } else {
        None
    };

    // Apply modifications if this is a request with a body that needs parameter adjustment
    let modified_body_bytes = if let Some(ref mut json) = body_json {
        if let Some(model_name) = extract_model_name(json) {
            info!("🔍 Detected model: {}", model_name);
            
            // Fetch model metadata
            match state.metadata_cache.get_model_info(&model_name).await {
                Ok(metadata) => {
                    info!("📊 Model metadata - n_ctx_train: {}", metadata.n_ctx_train);
                    
                    // Apply modifiers
                    let modified = apply_modifiers(json, &metadata, state.max_context_override);
                    if modified {
                        info!("✏️  Request modified - see changes above");
                    }
                }
                Err(e) => {
                    warn!("⚠️  Could not fetch model metadata: {}", e);
                }
            }
        }
        
        // Serialize the potentially modified JSON back to bytes
        serde_json::to_vec(json).unwrap_or_else(|_| body_bytes.to_vec())
    } else {
        body_bytes.to_vec()
    };

    // Build the proxied request
    let target_url = format!("{}{}", state.ollama_host, path);
    let full_url = if query.is_empty() {
        target_url
    } else {
        format!("{}?{}", target_url, query)
    };

    debug!("🔄 Forwarding to: {}", full_url);
    debug!("📦 Request body size: {} bytes", modified_body_bytes.len());
    
    // Log the actual body being sent for debugging
    if let Ok(body_str) = String::from_utf8(modified_body_bytes.clone()) {
        debug!("📤 Request body being sent to Ollama: {}", body_str);
    }

    // Create the proxied request
    let mut proxy_req = state.client
        .request(method.clone(), &full_url)
        .body(modified_body_bytes);

    // Copy headers, but skip host and content-length
    // (content-length will be set automatically by reqwest based on body)
    let mut has_content_type = false;
    for (key, value) in headers.iter() {
        let key_lower = key.as_str().to_lowercase();
        if key_lower == "content-type" {
            has_content_type = true;
        }
        if key_lower != "host" && key_lower != "content-length" {
            proxy_req = proxy_req.header(key, value);
        }
    }
    
    // Ensure Content-Type is set for JSON bodies
    if !has_content_type && body_json.is_some() {
        debug!("   Setting Content-Type: application/json");
        proxy_req = proxy_req.header("Content-Type", "application/json");
    }

    // Check if this is a streaming request (do this BEFORE sending)
    let is_streaming = is_streaming_request(&body_json);
    if is_streaming {
        info!("🌊 Streaming request detected - will forward chunks in real-time");
    } else {
        info!("📦 Non-streaming request - will buffer full response");
    }

    // Send the request
    info!("🚀 Sending request to Ollama (timeout: {}s)", state.request_timeout_seconds);
    debug!("📤 Awaiting response from Ollama...");
    let response = match proxy_req.send().await {
        Ok(resp) => {
            debug!("✓ Received response headers from Ollama");
            resp
        }
        Err(e) => {
            if e.is_timeout() {
                error!("⏱️  Request timed out after {} seconds", state.request_timeout_seconds);
                error!("   This usually indicates Ollama is stalled or processing very large context");
                error!("   Try: Reduce MAX_CONTEXT_OVERRIDE, restart Ollama, or check Ollama logs");
                return Err(StatusCode::GATEWAY_TIMEOUT);
            }
            error!("❌ Failed to proxy request: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = response.status();
    info!("📬 Response status: {}", status);

    // Only use streaming for successful responses (2xx)
    // Error responses (4xx, 5xx) are single JSON objects, not NDJSON streams
    if is_streaming && status.is_success() {
        info!("🌊 Forwarding response chunks in real-time");
        return stream_standard_response(response, status).await;
    } else if is_streaming && !status.is_success() {
        warn!("⚠️  Streaming requested but got error status {}, falling back to buffered response", status);
    }
    
    if !status.is_success() {
        debug!("📥 Reading error response body...");
    } else {
        debug!("📥 Reading response body...");
    }

    // Build response
    let mut builder = Response::builder().status(status);
    
    // Copy response headers
    for (key, value) in response.headers().iter() {
        builder = builder.header(key, value);
    }

    // Get response body
    let response_bytes = match response.bytes().await {
        Ok(bytes) => {
            debug!("✓ Read {} bytes from response body", bytes.len());
            bytes
        }
        Err(e) => {
            error!("❌ Failed to read response body: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    // Log response body if it's JSON and not too large
    if !response_bytes.is_empty() && response_bytes.len() < 10000 {
        if let Ok(json) = serde_json::from_slice::<Value>(&response_bytes) {
            if !status.is_success() {
                error!("❌ Ollama error response: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
            } else {
                debug!("📄 Response body: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
            }
        }
    }

    let body = Body::from(response_bytes);
    
    debug!("✓ Building response to send back to client");
    let result = builder.body(body).map_err(|e| {
        error!("Failed to build response: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    });
    
    if result.is_ok() {
        info!("✅ Successfully completed request - response sent to client");
    }
    result
}

/// Check if a request has streaming enabled
fn is_streaming_request(json: &Option<Value>) -> bool {
    let stream_value = json.as_ref().and_then(|j| j.get("stream"));
    let result = stream_value.and_then(|s| s.as_bool()).unwrap_or(false);
    debug!("🔍 Streaming check: stream={:?}, result={}", stream_value, result);
    result
}

/// Stream response from Ollama directly to client without buffering
async fn stream_standard_response(
    response: reqwest::Response,
    status: StatusCode,
) -> Result<Response<Body>, StatusCode> {
    use tokio_stream::wrappers::ReceiverStream;
    
    info!("🌊 Starting real-time NDJSON streaming");
    let start_time = std::time::Instant::now();
    
    let mut builder = Response::builder().status(status);
    
    // Copy response headers (especially Content-Type)
    for (key, value) in response.headers().iter() {
        builder = builder.header(key, value);
        debug!("   Header: {}: {:?}", key, value);
    }
    
    // Create bounded channel for chunk forwarding (capacity 100)
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(100);
    
    // Spawn background task to process Ollama's stream
    tokio::spawn(async move {
        if let Err(e) = process_streaming_chunks(response, tx, start_time).await {
            error!("❌ Streaming task failed: {}", e);
        }
    });
    
    // Create response body from channel receiver
    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    
    builder.body(body).map_err(|e| {
        error!("Failed to build streaming response: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

/// Process streaming chunks from Ollama, forwarding complete NDJSON lines immediately
async fn process_streaming_chunks(
    response: reqwest::Response,
    tx: tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
    start_time: std::time::Instant,
) -> Result<(), String> {
    use futures::StreamExt;
    
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut chunk_count = 0;
    let mut total_bytes = 0;
    let mut lines_forwarded = 0;
    
    info!("📡 Stream processor started, waiting for chunks from Ollama...");
    
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                chunk_count += 1;
                let chunk_size = chunk.len();
                total_bytes += chunk_size;
                let elapsed = start_time.elapsed();
                
                debug!("📦 Chunk #{} received: {} bytes at {:?}", chunk_count, chunk_size, elapsed);
                
                // Add chunk to buffer
                buffer.extend_from_slice(&chunk);
                
                // Process complete lines from buffer
                loop {
                    if let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                        // Extract complete line (including newline)
                        let line_bytes = buffer.drain(..=newline_pos).collect::<Vec<u8>>();
                        let line_len = line_bytes.len();
                        
                        lines_forwarded += 1;
                        debug!("✉️  Forwarding line #{}: {} bytes", lines_forwarded, line_len);
                        
                        // Forward line to client immediately
                        let send_result = tx.send(Ok(bytes::Bytes::from(line_bytes))).await;
                        
                        match send_result {
                            Ok(_) => {
                                debug!("✓ Line #{} forwarded successfully", lines_forwarded);
                            }
                            Err(_) => {
                                // Channel closed, client disconnected
                                warn!("⚠️  Client disconnected (channel closed) after {} lines", lines_forwarded);
                                return Err("Client disconnected".to_string());
                            }
                        }
                    } else {
                        // No complete line yet, wait for more data
                        debug!("⏳ Partial line in buffer ({} bytes), waiting for more data", buffer.len());
                        break;
                    }
                }
            }
            Err(e) => {
                error!("❌ Stream error on chunk #{}: {}", chunk_count + 1, e);
                
                // Don't break on transient errors, log and continue
                if e.is_timeout() {
                    error!("   Timeout error - this may indicate Ollama is stalled");
                } else if e.is_connect() {
                    error!("   Connection error - Ollama may have disconnected");
                    return Err(format!("Connection error: {}", e));
                } else {
                    warn!("   Transient error, continuing stream: {}", e);
                }
            }
        }
    }
    
    // Stream ended, check for remaining data in buffer
    if !buffer.is_empty() {
        warn!("⚠️  Stream ended with {} bytes remaining in buffer (incomplete line)", buffer.len());
        
        // Forward remaining bytes if any (incomplete final line)
        if tx.send(Ok(bytes::Bytes::from(buffer))).await.is_err() {
            warn!("   Failed to forward remaining bytes, client disconnected");
        }
    }
    
    let elapsed = start_time.elapsed();
    info!("✅ Stream completed successfully:");
    info!("   Total chunks: {}", chunk_count);
    info!("   Total bytes: {}", total_bytes);
    info!("   Lines forwarded: {}", lines_forwarded);
    info!("   Duration: {:?}", elapsed);
    info!("   Throughput: {:.2} KB/s", (total_bytes as f64 / 1024.0) / elapsed.as_secs_f64());
    
    Ok(())
}

fn extract_model_name(json: &Value) -> Option<String> {
    // Try OpenAI API format first
    if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
        return Some(model.to_string());
    }
    
    // Try Ollama API format
    if let Some(model) = json.get("name").and_then(|v| v.as_str()) {
        return Some(model.to_string());
    }
    
    None
}

