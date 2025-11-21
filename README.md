# Ollama Proxy

A lightweight Rust proxy for Ollama that intelligently adjusts request parameters to match each model's training configuration.

## Problem

Some AI clients (like Elephas) send the same context length parameter for all models. This causes issues when:

- Embedding models trained with 8K context receive requests for 128K context
- Ollama warns: "requested context size too large for model"
- Models may perform poorly with incorrect parameters

## Solution

This proxy sits between your client and Ollama, automatically:

- Detects which model is being requested
- Fetches the model's training context length (`n_ctx_train`)
- Adjusts `num_ctx` if it exceeds the model's capabilities
- Provides detailed logging of all modifications

## Features

- ✅ Automatic parameter correction based on model metadata
- ✅ Request/response logging for debugging
- ✅ Model metadata caching for performance
- ✅ Extensible modifier framework for future enhancements
- ✅ Zero configuration for basic usage

## Installation

```bash
cargo build --release
```

## Usage

### 1. Start the Proxy

```bash
# Default: Listen on 127.0.0.1:11435, proxy to 127.0.0.1:11434
cargo run --release

# Or with custom settings:
OLLAMA_HOST=http://127.0.0.1:11434 PROXY_PORT=11435 RUST_LOG=info cargo run --release
```

### 2. Configure Your Client

Point your AI client (Elephas, etc.) to the proxy instead of Ollama directly:

**Before:** `http://127.0.0.1:11434`  
**After:** `http://127.0.0.1:11435`

### 3. Watch the Magic

The proxy will log all requests and modifications:

```
📨 Incoming request: POST /v1/embeddings
📋 Request body: {
  "model": "nomic-embed-text",
  "input": "test",
  "options": {
    "num_ctx": 131072
  }
}
🔍 Detected model: nomic-embed-text
📊 Model metadata - n_ctx_train: 8192
⚠️  num_ctx (131072) exceeds model training context (8192)
✏️  Modified options.num_ctx: 131072 → 8192
🔧 ContextLimitModifier applied modifications
📬 Response status: 200 OK
```

## Configuration

Environment variables:

- `OLLAMA_HOST` - Target Ollama server (default: `http://127.0.0.1:11434`)
- `PROXY_PORT` - Port to listen on (default: `11435`)
- `RUST_LOG` - Log level: `error`, `warn`, `info`, `debug`, `trace` (default: `info`)

## How It Works

1. **Intercept**: Proxy receives request from client
2. **Detect API Format**: Determine if request uses OpenAI or native Ollama API
3. **Translate** (if needed): Convert OpenAI `/v1/embeddings` → Ollama `/api/embed`
4. **Fetch Metadata**: Query Ollama API for model's training parameters
5. **Inject Parameters**: Add `options.num_ctx` with correct value for the model
6. **Forward**: Send request to Ollama native API (which accepts options)
7. **Translate Response**: Convert Ollama response back to OpenAI format
8. **Return**: Pass OpenAI-compatible response back to client

## Architecture

```
Client (Elephas)
    ↓ OpenAI API format (/v1/embeddings)
Proxy (Port 11435)
    ↓ Translates to native Ollama API (/api/embed)
    ↓ Injects options.num_ctx based on model
Ollama (Port 11434)
    ↓ Returns native response
Proxy
    ↓ Translates back to OpenAI format
Client receives OpenAI-compatible response
```

**Key Innovation**: The proxy acts as a translation layer, converting between OpenAI's API format (which doesn't support runtime options) and Ollama's native API (which does), enabling per-request parameter control without changing global settings.

## Extending

The modifier framework is designed for easy extension:

```rust
pub trait ParameterModifier {
    fn modify(&self, json: &mut Value, metadata: &ModelMetadata) -> bool;
    fn name(&self) -> &str;
}
```

Add new modifiers in `src/modifier.rs` and register them in `apply_modifiers()`.

## Testing

```bash
cargo test
```

## License

MIT
