use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub n_ctx_train: u32,
    pub model_type: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            n_ctx_train: 8192, // Reasonable default
            model_type: "unknown".to_string(),
        }
    }
}

pub struct ModelMetadataCache {
    cache: Mutex<HashMap<String, ModelMetadata>>,
    ollama_host: String,
    client: reqwest::Client,
}

impl ModelMetadataCache {
    pub fn new(ollama_host: String) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            ollama_host,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_model_info(&self, model_name: &str) -> Result<ModelMetadata, String> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(metadata) = cache.get(model_name) {
                debug!("Cache hit for model: {}", model_name);
                return Ok(metadata.clone());
            }
        }

        debug!("Cache miss for model: {}, fetching from Ollama API", model_name);

        // Fetch from Ollama API
        let metadata = self.fetch_model_info(model_name).await?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(model_name.to_string(), metadata.clone());
        }

        Ok(metadata)
    }

    async fn fetch_model_info(&self, model_name: &str) -> Result<ModelMetadata, String> {
        let url = format!("{}/api/show", self.ollama_host);
        
        let request_body = serde_json::json!({
            "name": model_name
        });

        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch model info: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Ollama API returned error: {}", response.status()));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        // Extract n_ctx_train from model details
        let n_ctx_train = self.extract_n_ctx_train(&response_json);
        let model_type = self.extract_model_type(&response_json);

        Ok(ModelMetadata {
            n_ctx_train,
            model_type,
        })
    }

    fn extract_n_ctx_train(&self, response: &serde_json::Value) -> u32 {
        // Try to extract from model_info -> llama.context_length or similar fields
        // The response structure may vary, so we'll try multiple paths
        
        // Check modelinfo first
        if let Some(model_info) = response.get("model_info") {
            if let Some(params) = model_info.as_object() {
                // Try various possible field names
                for key in params.keys() {
                    if key.contains("context") || key.contains("ctx") {
                        if let Some(value) = params.get(key).and_then(|v| v.as_u64()) {
                            debug!("Found n_ctx_train in model_info.{}: {}", key, value);
                            return value as u32;
                        }
                    }
                }
            }
        }

        // Check details field
        if let Some(details) = response.get("details") {
            if let Some(params) = details.get("parameters") {
                if let Some(n_ctx) = params.get("num_ctx").and_then(|v| v.as_u64()) {
                    debug!("Found n_ctx in details.parameters: {}", n_ctx);
                    return n_ctx as u32;
                }
            }
        }

        // Try parsing the modelfile for context information
        if let Some(modelfile) = response.get("modelfile").and_then(|v| v.as_str()) {
            if let Some(ctx) = self.extract_ctx_from_modelfile(modelfile) {
                debug!("Found context in modelfile: {}", ctx);
                return ctx;
            }
        }

        // Try template or parameters
        if let Some(parameters) = response.get("parameters") {
            if let Some(params_str) = parameters.as_str() {
                if let Some(ctx) = self.extract_ctx_from_params(params_str) {
                    debug!("Found context in parameters: {}", ctx);
                    return ctx;
                }
            }
        }

        warn!("Could not find n_ctx_train in model info, using default 8192");
        warn!("Response structure: {}", serde_json::to_string_pretty(response).unwrap_or_default());
        
        8192 // Default fallback
    }

    fn extract_model_type(&self, response: &serde_json::Value) -> String {
        // Check if this is an embedding model
        if let Some(modelfile) = response.get("modelfile").and_then(|v| v.as_str()) {
            if modelfile.to_lowercase().contains("embed") {
                return "embedding".to_string();
            }
        }

        if let Some(template) = response.get("template").and_then(|v| v.as_str()) {
            if template.is_empty() || template.contains("{{ .Prompt }}") {
                return "embedding".to_string();
            }
        }

        "chat".to_string()
    }

    fn extract_ctx_from_modelfile(&self, modelfile: &str) -> Option<u32> {
        // Look for PARAMETER num_ctx in the modelfile
        for line in modelfile.lines() {
            if line.to_lowercase().contains("parameter") && line.contains("num_ctx") {
                // Extract the number
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(last) = parts.last() {
                    if let Ok(value) = last.parse::<u32>() {
                        return Some(value);
                    }
                }
            }
        }
        None
    }

    fn extract_ctx_from_params(&self, params: &str) -> Option<u32> {
        // Look for num_ctx in a parameter string
        for line in params.lines() {
            if line.contains("num_ctx") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(last) = parts.last() {
                    if let Ok(value) = last.parse::<u32>() {
                        return Some(value);
                    }
                }
            }
        }
        None
    }
}

