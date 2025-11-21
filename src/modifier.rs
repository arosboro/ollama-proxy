use serde_json::Value;
use tracing::{info, warn};
use crate::model_metadata::ModelMetadata;

/// Trait for parameter modifiers
/// Each modifier can inspect and modify request parameters
pub trait ParameterModifier {
    fn modify(&self, json: &mut Value, metadata: &ModelMetadata, max_context_override: u32) -> bool;
    fn name(&self) -> &str;
}

/// Num predict modifier - adds num_predict to prevent infinite generation
pub struct NumPredictModifier;

impl ParameterModifier for NumPredictModifier {
    fn modify(&self, json: &mut Value, _metadata: &ModelMetadata, _max_context_override: u32) -> bool {
        let mut modified = false;

        // Only apply to chat requests (not embeddings)
        let is_chat = json.get("messages").is_some();
        if !is_chat {
            return false;
        }

        // Check if num_predict already exists
        let has_num_predict = json.get("options")
            .and_then(|o| o.get("num_predict"))
            .is_some();

        if has_num_predict {
            info!("✅ num_predict already set");
            return false;
        }

        // Get max_tokens from request (OpenAI format) or use default
        let max_tokens = json.get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as u32;

        info!("ℹ️  No num_predict specified, adding to prevent infinite generation");
        
        // Ensure options object exists
        if json.get("options").is_none() {
            if let Some(obj) = json.as_object_mut() {
                obj.insert("options".to_string(), Value::Object(Default::default()));
            }
        }

        // Add num_predict to options
        if let Some(options) = json.get_mut("options") {
            if let Some(options_obj) = options.as_object_mut() {
                options_obj.insert(
                    "num_predict".to_string(),
                    Value::Number(max_tokens.into())
                );
                info!(
                    "✏️  Added options.num_predict: {} (from max_tokens or default)",
                    max_tokens
                );
                modified = true;
            }
        }

        modified
    }

    fn name(&self) -> &str {
        "NumPredictModifier"
    }
}

/// Context limit modifier - ensures num_ctx doesn't exceed model's training context or override limit
pub struct ContextLimitModifier;

impl ParameterModifier for ContextLimitModifier {
    fn modify(&self, json: &mut Value, metadata: &ModelMetadata, max_context_override: u32) -> bool {
        let mut modified = false;

        // Determine the effective maximum: min(model's n_ctx_train, max_context_override)
        let effective_max = metadata.n_ctx_train.min(max_context_override);
        
        // Log decision rationale clearly
        info!("📊 Context size decision:");
        info!("   Model capability (n_ctx_train): {}", metadata.n_ctx_train);
        info!("   User override (MAX_CONTEXT_OVERRIDE): {}", max_context_override);
        info!("   Effective limit: {}", effective_max);
        
        if effective_max < metadata.n_ctx_train {
            info!(
                "   ℹ️  Using override limit ({}) instead of model's full capacity ({}) for stability",
                effective_max, metadata.n_ctx_train
            );
        } else if effective_max == metadata.n_ctx_train {
            info!(
                "   ℹ️  Using model's native capacity ({})",
                effective_max
            );
        }

        // Issue warnings for potentially problematic context sizes
        Self::warn_on_large_context(effective_max);

        // Check options.num_ctx (Ollama native format)
        if let Some(options) = json.get_mut("options") {
            if let Some(options_obj) = options.as_object_mut() {
                if let Some(num_ctx) = options_obj.get("num_ctx") {
                    if let Some(current_ctx) = num_ctx.as_u64() {
                        if current_ctx > effective_max as u64 {
                            warn!(
                                "⚠️  num_ctx ({}) exceeds safe limit ({})",
                                current_ctx, effective_max
                            );
                            options_obj.insert(
                                "num_ctx".to_string(),
                                Value::Number(effective_max.into())
                            );
                            info!(
                                "✏️  Modified options.num_ctx: {} → {}",
                                current_ctx, effective_max
                            );
                            modified = true;
                        } else {
                            info!(
                                "✅ num_ctx ({}) is within safe limits ({})",
                                current_ctx, effective_max
                            );
                        }
                    }
                }
            }
        }

        // Check top-level num_ctx (alternative format)
        if let Some(num_ctx) = json.get("num_ctx") {
            if let Some(current_ctx) = num_ctx.as_u64() {
                if current_ctx > effective_max as u64 {
                    warn!(
                        "⚠️  num_ctx ({}) exceeds safe limit ({})",
                        current_ctx, effective_max
                    );
                    if let Some(obj) = json.as_object_mut() {
                        obj.insert(
                            "num_ctx".to_string(),
                            Value::Number(effective_max.into())
                        );
                        info!(
                            "✏️  Modified num_ctx: {} → {}",
                            current_ctx, effective_max
                        );
                        modified = true;
                    }
                } else {
                    info!(
                        "✅ num_ctx ({}) is within safe limits ({})",
                        current_ctx, effective_max
                    );
                }
            }
        }

        // If num_ctx wasn't present, we should set it explicitly to prevent
        // Ollama from using the global OLLAMA_CONTEXT_LENGTH setting or model defaults
        if !modified && !json.get("num_ctx").is_some() {
            // Check if options object exists
            let has_options_num_ctx = json.get("options")
                .and_then(|o| o.get("num_ctx"))
                .is_some();
            
            if !has_options_num_ctx {
                // For chat models: always set to effective_max
                // For embedding models: only set if it's at or below the model's natural capacity
                let should_set_ctx = if metadata.model_type == "chat" {
                    true
                } else {
                    // For embeddings, only set if override doesn't exceed model's natural limit
                    effective_max <= metadata.n_ctx_train
                };
                
                if should_set_ctx {
                    info!(
                        "ℹ️  No num_ctx specified, setting to effective limit: {}",
                        effective_max
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
                                Value::Number(effective_max.into())
                            );
                            info!(
                                "✏️  Added options.num_ctx: {} (model_type: {})",
                                effective_max, metadata.model_type
                            );
                            modified = true;
                        }
                    }
                } else {
                    info!(
                        "ℹ️  Skipping num_ctx for {} model (override {} > model capacity {})",
                        metadata.model_type, effective_max, metadata.n_ctx_train
                    );
                }
            }
        }

        modified
    }

    fn name(&self) -> &str {
        "ContextLimitModifier"
    }
}

impl ContextLimitModifier {
    /// Warn users about potentially problematic context sizes
    fn warn_on_large_context(ctx_size: u32) {
        match ctx_size {
            100001..=u32::MAX => {
                warn!("⚠️⚠️⚠️  CRITICAL: Context size {} > 100K tokens - Very likely to stall!", ctx_size);
                warn!("   Recommendation: Reduce to 16K-32K for reliable operation");
            }
            60001..=100000 => {
                warn!("⚠️⚠️  WARNING: Context size {} > 60K tokens - May cause stalls", ctx_size);
                warn!("   Recommendation: Monitor for timeouts, consider reducing to 32K");
            }
            32001..=60000 => {
                warn!("⚠️  CAUTION: Context size {} > 32K tokens - Slower response expected", ctx_size);
                warn!("   This may work but could be unstable with flash attention");
            }
            16385..=32000 => {
                info!("ℹ️  Context size {} is moderate - Should work reliably", ctx_size);
            }
            _ => {
                // 16K or less - optimal range, no warning needed
            }
        }
    }
}

/// Apply all modifiers to the request
pub fn apply_modifiers(json: &mut Value, metadata: &ModelMetadata, max_context_override: u32) -> bool {
    let modifiers: Vec<Box<dyn ParameterModifier>> = vec![
        Box::new(NumPredictModifier),  // Must run first to prevent infinite generation
        Box::new(ContextLimitModifier),
        // Future modifiers can be added here
    ];

    let mut any_modified = false;

    for modifier in modifiers {
        if modifier.modify(json, metadata, max_context_override) {
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
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            8192 // Model limit is lower than override
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
        let modified = modifier.modify(&mut request, &metadata, 16384);

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
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            8192
        );
    }

    #[test]
    fn test_adds_num_ctx_for_chat_without_it() {
        let mut request = json!({
            "model": "llama3.3",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = ContextLimitModifier;
        // Override caps the model's capability
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        // Should set to override limit (16384) not model's full capacity (131072)
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            16384
        );
    }

    #[test]
    fn test_max_context_override_caps_large_model() {
        let mut request = json!({
            "model": "gpt-oss:20b",
            "options": {
                "num_ctx": 131072
            }
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = ContextLimitModifier;
        // Override caps at 16384
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        assert_eq!(
            request["options"]["num_ctx"].as_u64().unwrap(),
            16384 // Capped by override, not model limit
        );
    }

    #[test]
    fn test_num_predict_added_when_missing() {
        let mut request = json!({
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = NumPredictModifier;
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        assert_eq!(
            request["options"]["num_predict"].as_u64().unwrap(),
            4096 // Default value
        );
    }

    #[test]
    fn test_num_predict_uses_max_tokens() {
        let mut request = json!({
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 2048
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = NumPredictModifier;
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(modified);
        assert_eq!(
            request["options"]["num_predict"].as_u64().unwrap(),
            2048 // Uses max_tokens value
        );
    }

    #[test]
    fn test_num_predict_preserved_when_exists() {
        let mut request = json!({
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "options": {
                "num_predict": 1000
            }
        });

        let metadata = ModelMetadata {
            n_ctx_train: 131072,
            model_type: "chat".to_string(),
        };

        let modifier = NumPredictModifier;
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(!modified); // Should not modify
        assert_eq!(
            request["options"]["num_predict"].as_u64().unwrap(),
            1000 // Preserved existing value
        );
    }

    #[test]
    fn test_num_predict_not_added_to_embeddings() {
        let mut request = json!({
            "model": "nomic-embed-text",
            "input": "test text"
        });

        let metadata = ModelMetadata {
            n_ctx_train: 8192,
            model_type: "embedding".to_string(),
        };

        let modifier = NumPredictModifier;
        let modified = modifier.modify(&mut request, &metadata, 16384);

        assert!(!modified); // Should not modify embeddings
        assert!(request.get("options").is_none());
    }
}


