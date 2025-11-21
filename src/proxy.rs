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
    let path = uri.path();
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

