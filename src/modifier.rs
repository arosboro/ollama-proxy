use serde_json::Value;
use tracing::{info, warn};
use crate::model_metadata::ModelMetadata;

/// Trait for parameter modifiers
/// Each modifier can inspect and modify request parameters
pub trait ParameterModifier {
    fn modify(&self, json: &mut Value, metadata: &ModelMetadata) -> bool;
    fn name(&self) -> &str;
}

/// Context limit modifier - ensures num_ctx doesn't exceed model's training context
pub struct ContextLimitModifier;

impl ParameterModifier for ContextLimitModifier {
    fn modify(&self, json: &mut Value, metadata: &ModelMetadata) -> bool {
        let mut modified = false;

        // Check options.num_ctx (Ollama native format)
        if let Some(options) = json.get_mut("options") {
            if let Some(options_obj) = options.as_object_mut() {
                if let Some(num_ctx) = options_obj.get("num_ctx") {
                    if let Some(current_ctx) = num_ctx.as_u64() {
                        if current_ctx > metadata.n_ctx_train as u64 {
                            warn!(
                                "⚠️  num_ctx ({}) exceeds model training context ({})",
                                current_ctx, metadata.n_ctx_train
                            );
                            options_obj.insert(
                                "num_ctx".to_string(),
                                Value::Number(metadata.n_ctx_train.into())
                            );
                            info!(
                                "✏️  Modified options.num_ctx: {} → {}",
                                current_ctx, metadata.n_ctx_train
                            );
                            modified = true;
                        } else {
                            info!(
                                "✅ num_ctx ({}) is within model limits ({})",
                                current_ctx, metadata.n_ctx_train
                            );
                        }
                    }
                }
            }
        }

        // Check top-level num_ctx (alternative format)
        if let Some(num_ctx) = json.get("num_ctx") {
            if let Some(current_ctx) = num_ctx.as_u64() {
                if current_ctx > metadata.n_ctx_train as u64 {
                    warn!(
                        "⚠️  num_ctx ({}) exceeds model training context ({})",
                        current_ctx, metadata.n_ctx_train
                    );
                    if let Some(obj) = json.as_object_mut() {
                        obj.insert(
                            "num_ctx".to_string(),
                            Value::Number(metadata.n_ctx_train.into())
                        );
                        info!(
                            "✏️  Modified num_ctx: {} → {}",
                            current_ctx, metadata.n_ctx_train
                        );
                        modified = true;
                    }
                } else {
                    info!(
                        "✅ num_ctx ({}) is within model limits ({})",
                        current_ctx, metadata.n_ctx_train
                    );
                }
            }
        }

        // If num_ctx wasn't present but we're dealing with an embedding request,
        // we might want to set it explicitly to prevent Ollama from using the
        // global OLLAMA_CONTEXT_LENGTH setting
        if !modified && !json.get("num_ctx").is_some() {
            // Check if options object exists
            let has_options_num_ctx = json.get("options")
                .and_then(|o| o.get("num_ctx"))
                .is_some();
            
            if !has_options_num_ctx {
                // This is a request without explicit num_ctx
                // For embedding models, we should set it to avoid using the global setting
                if metadata.model_type == "embedding" {
                    info!(
                        "ℹ️  No num_ctx specified for embedding model, setting to model limit: {}",
                        metadata.n_ctx_train
                    );
                    
                    // Ensure options object exists
                    if !json.get("options").is_some() {
                        if let Some(obj) = json.as_object_mut() {
                            obj.insert("options".to_string(), Value::Object(Default::default()));
                        }
                    }
                    
                    // Set num_ctx in options
                    if let Some(options) = json.get_mut("options") {
                        if let Some(options_obj) = options.as_object_mut() {
                            options_obj.insert(
                                "num_ctx".to_string(),
                                Value::Number(metadata.n_ctx_train.into())
                            );
                            info!(
                                "✏️  Added options.num_ctx: {}",
                                metadata.n_ctx_train
                            );
                            modified = true;
                        }
                    }
                }
            }
        }

        modified
    }

    fn name(&self) -> &str {
        "ContextLimitModifier"
    }
}

/// Apply all modifiers to the request
pub fn apply_modifiers(json: &mut Value, metadata: &ModelMetadata) -> bool {
    let modifiers: Vec<Box<dyn ParameterModifier>> = vec![
        Box::new(ContextLimitModifier),
        // Future modifiers can be added here
    ];

    let mut any_modified = false;

    for modifier in modifiers {
        if modifier.modify(json, metadata) {
            info!("🔧 {} applied modifications", modifier.name());
            any_modified = true;
        }
    }

    any_modified
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_context_limit_modifier_reduces_high_value() {
        let mut request = json!({
            "model": "nomic-embed-text",
            "options": {
                "num_ctx": 131072
            }
        });

        let metadata = ModelMetadata {
            n_ctx_train: 8192,
            model_type: "embedding".to_string(),
        };

        let modifier = ContextLimitModifier;
        let modified = modifier.modify(&mut request, &metadata);

        assert!(modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            8192
        );
    }

    #[test]
    fn test_context_limit_modifier_keeps_valid_value() {
        let mut request = json!({
            "model": "nomic-embed-text",
            "options": {
                "num_ctx": 4096
            }
        });

        let metadata = ModelMetadata {
            n_ctx_train: 8192,
            model_type: "embedding".to_string(),
        };

        let modifier = ContextLimitModifier;
        let modified = modifier.modify(&mut request, &metadata);

        assert!(!modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            4096
        );
    }

    #[test]
    fn test_adds_num_ctx_for_embedding_without_it() {
        let mut request = json!({
            "model": "nomic-embed-text",
            "input": "test text"
        });

        let metadata = ModelMetadata {
            n_ctx_train: 8192,
            model_type: "embedding".to_string(),
        };

        let modifier = ContextLimitModifier;
        let modified = modifier.modify(&mut request, &metadata);

        assert!(modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            8192
        );
    }

    #[test]
    fn test_does_not_add_num_ctx_for_chat_without_it() {
        let mut request = json!({
            "model": "llama3.3",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = ContextLimitModifier;
        let modified = modifier.modify(&mut request, &metadata);

        assert!(!modified);
        assert!(request.get("options").is_none());
    }
}

