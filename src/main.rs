mod proxy;
mod model_metadata;
mod modifier;
mod translator;
mod chunker;

use axum::{Router, serve};
use std::env;
use tokio::net::TcpListener;
use tracing::{info, Level};

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(
            env::var("RUST_LOG")
                .ok()
                .and_then(|s| s.parse::<Level>().ok())
                .unwrap_or(Level::INFO)
        )
        .init();

    // Configuration from environment variables
    let ollama_host = env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let proxy_port = env::var("PROXY_PORT")
        .unwrap_or_else(|_| "11435".to_string());
    let bind_addr = format!("127.0.0.1:{}", proxy_port);

    // Chunking configuration
    let max_embedding_input_length = env::var("MAX_EMBEDDING_INPUT_LENGTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);
    
    let enable_auto_chunking = env::var("ENABLE_AUTO_CHUNKING")
        .ok()
        .map(|s| s.to_lowercase() != "false" && s != "0")
        .unwrap_or(true);

    // Context override configuration (prevents large context stalls)
    let max_context_override = env::var("MAX_CONTEXT_OVERRIDE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(16384);

    // Timeout configuration (prevents indefinite hangs)
    let request_timeout_seconds = env::var("REQUEST_TIMEOUT_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(120);

    info!("Starting Ollama Proxy");
    info!("Listening on: {}", bind_addr);
    info!("Proxying to: {}", ollama_host);
    info!("Chunking config:");
    info!("  Max embedding input length: {}", max_embedding_input_length);
    info!("  Auto chunking enabled: {}", enable_auto_chunking);
    info!("Context config:");
    info!("  Max context override: {} (hard cap for stability)", max_context_override);
    info!("  Request timeout: {} seconds", request_timeout_seconds);

    // Validate configuration
    if max_embedding_input_length < 100 {
        panic!("MAX_EMBEDDING_INPUT_LENGTH must be at least 100 characters");
    }
    if max_context_override < 512 {
        panic!("MAX_CONTEXT_OVERRIDE must be at least 512 tokens");
    }

    // Create shared state
    let state = proxy::ProxyState::new(
        ollama_host,
        max_embedding_input_length,
        enable_auto_chunking,
        max_context_override,
        request_timeout_seconds,
    );

    // Build router
    let app = Router::new()
        .fallback(proxy::proxy_handler)
        .with_state(state);

    // Start server
    let listener = TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind to address");
    
    info!("Ollama Proxy is ready");
    
    serve(listener, app)
        .await
        .expect("Server error");
}

