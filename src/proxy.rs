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
};

#[derive(Clone)]
pub struct ProxyState {
    pub ollama_host: String,
    pub client: reqwest::Client,
    pub metadata_cache: Arc<ModelMetadataCache>,
}

impl ProxyState {
    pub fn new(ollama_host: String) -> Self {
        Self {
            ollama_host: ollama_host.clone(),
            client: reqwest::Client::new(),
            metadata_cache: Arc::new(ModelMetadataCache::new(ollama_host)),
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
    headers: axum::http::HeaderMap,
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

    // Get target endpoint
    let target_path = get_ollama_endpoint(path);
    
    // Translate based on endpoint
    let (translated_body, original_model) = match path {
        "/v1/embeddings" => {
            let ollama_req = match translate_openai_embeddings_to_ollama(body_json, metadata.n_ctx_train) {
                Ok(req) => req,
                Err(e) => {
                    error!("Failed to translate request: {}", e);
                    return Err(StatusCode::BAD_REQUEST);
                }
            };
            
            let model = ollama_req.model.clone();
            let body = match serde_json::to_vec(&ollama_req) {
                Ok(b) => b,
                Err(e) => {
                    error!("Failed to serialize translated request: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };
            
            info!("📤 Translated request: {}", serde_json::to_string_pretty(&ollama_req).unwrap_or_default());
            
            (body, model)
        }
        _ => {
            error!("Translation not implemented for path: {}", path);
            return Err(StatusCode::NOT_IMPLEMENTED);
        }
    };

    // Send request to Ollama native API
    let target_url = format!("{}{}", state.ollama_host, target_path);
    info!("🔄 Forwarding to Ollama native API: {}", target_url);

    let mut proxy_req = state.client
        .post(&target_url)
        .body(translated_body)
        .header("Content-Type", "application/json");

    // Copy relevant headers
    for (key, value) in headers.iter() {
        if key != "host" && key != "content-length" {
            proxy_req = proxy_req.header(key, value);
        }
    }

    // Send the request
    let response = match proxy_req.send().await {
        Ok(resp) => resp,
        Err(e) => {
            error!("❌ Failed to proxy request: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = response.status();
    info!("📬 Ollama response status: {}", status);

    if !status.is_success() {
        error!("Ollama returned error status: {}", status);
        let error_body = response.bytes().await.unwrap_or_default();
        return Ok(Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(Body::from(error_body))
            .unwrap());
    }

    // Get response body
    let response_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read response body: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    // Parse Ollama response
    let ollama_resp: Value = match serde_json::from_slice(&response_bytes) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to parse Ollama response: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    debug!("📥 Ollama response: {}", serde_json::to_string_pretty(&ollama_resp).unwrap_or_default());

    // Translate response back to OpenAI format
    let openai_resp = match path {
        "/v1/embeddings" => {
            match translate_ollama_embed_to_openai(ollama_resp, original_model) {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Failed to translate response: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }
        _ => {
            error!("Response translation not implemented for path: {}", path);
            return Err(StatusCode::NOT_IMPLEMENTED);
        }
    };

    info!("✅ Translated response back to OpenAI format");

    // Serialize and return
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
                    let modified = apply_modifiers(json, &metadata);
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

    // Create the proxied request
    let mut proxy_req = state.client
        .request(method.clone(), &full_url)
        .body(modified_body_bytes);

    // Copy headers
    for (key, value) in headers.iter() {
        if key != "host" {
            proxy_req = proxy_req.header(key, value);
        }
    }

    // Send the request
    let response = match proxy_req.send().await {
        Ok(resp) => resp,
        Err(e) => {
            error!("❌ Failed to proxy request: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = response.status();
    info!("📬 Response status: {}", status);

    // Build response
    let mut builder = Response::builder().status(status);
    
    // Copy response headers
    for (key, value) in response.headers().iter() {
        builder = builder.header(key, value);
    }

    // Get response body
    let response_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read response body: {}", e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    // Log response body if it's JSON and not too large
    if !response_bytes.is_empty() && response_bytes.len() < 10000 {
        if let Ok(json) = serde_json::from_slice::<Value>(&response_bytes) {
            debug!("📄 Response body: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
        }
    }

    let body = Body::from(response_bytes);
    
    builder.body(body).map_err(|e| {
        error!("Failed to build response: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
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

