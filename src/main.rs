mod proxy;
mod model_metadata;
mod modifier;

use axum::{Router, serve};
use std::env;
use tokio::net::TcpListener;
use tracing::{info, Level};
use tracing_subscriber;

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

    info!("Starting Ollama Proxy");
    info!("Listening on: {}", bind_addr);
    info!("Proxying to: {}", ollama_host);

    // Create shared state
    let state = proxy::ProxyState::new(ollama_host);

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

