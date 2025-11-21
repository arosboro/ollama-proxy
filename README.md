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

- ✅ **Prevents infinite generation** - Auto-injects `num_predict` to limit output
- ✅ **Smart chunking** - Automatically splits large embeddings inputs to prevent crashes
- ✅ **Context safety caps** - Configurable hard limits to prevent Ollama stalls
- ✅ **Request timeouts** - Prevents indefinite hangs with configurable limits
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

### Context Size Configuration

**Prevent Ollama stalls with large contexts:**

- `MAX_CONTEXT_OVERRIDE` - Hard cap for context size regardless of model support (default: `16384`)
- `REQUEST_TIMEOUT_SECONDS` - Timeout for requests to Ollama (default: `120`)

**Why This Matters:**

Models may claim to support very large contexts (e.g., 131K tokens), but Ollama can stall or hang when actually processing them, especially with flash attention enabled. The `MAX_CONTEXT_OVERRIDE` provides a safety limit.

**Recommended Settings:**

```bash
# Conservative (most reliable)
MAX_CONTEXT_OVERRIDE=16384 REQUEST_TIMEOUT_SECONDS=120 cargo run --release

# Moderate (test with your hardware)
MAX_CONTEXT_OVERRIDE=32768 REQUEST_TIMEOUT_SECONDS=180 cargo run --release

# Aggressive (may cause stalls on some systems)
MAX_CONTEXT_OVERRIDE=65536 REQUEST_TIMEOUT_SECONDS=300 cargo run --release
```

**Note:** If requests time out, reduce `MAX_CONTEXT_OVERRIDE` first before increasing timeout.

### Generation Limit (num_predict)

**THE CRITICAL FIX FOR TIMEOUTS:**

The proxy automatically injects `num_predict` into all chat requests to prevent infinite generation loops.

**The Problem:**

- Ollama's default `num_predict` is **-1 (infinite)**
- Without this parameter, models generate until they fill entire context
- This causes "stalls" even with small contexts (4K)
- The model isn't stuck - it's generating millions of unwanted tokens!

**How the Proxy Fixes This:**

1. Detects chat requests (those with `messages` array)
2. Checks if `num_predict` is already set
3. If not set, injects `num_predict`:
   - Uses `max_tokens` from request if available (e.g., 4096 from Elephas)
   - Otherwise defaults to 4096 tokens
4. Logs the injection for transparency

**Example:**

```json
// Your request:
{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 2048
}

// Proxy automatically adds:
{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 2048,
  "options": {
    "num_predict": 2048  // ← Added by proxy
  }
}
```

**Why This Matters:**

Without `num_predict`, a simple "say hello" request can generate for 3+ minutes, filling the entire context buffer with elaborations, examples, and repetitions until it crashes or times out.

**Override if Needed:**

If you want different generation limits, set `num_predict` explicitly in your request - the proxy preserves existing values.

### Chunking Configuration

For large embeddings inputs, the proxy can automatically chunk text to prevent Ollama memory errors:

- `MAX_EMBEDDING_INPUT_LENGTH` - Maximum characters per embedding input (default: `2000`)
- `ENABLE_AUTO_CHUNKING` - Enable automatic chunking for large inputs (default: `true`)

**How Chunking Works:**

When an embeddings request contains text longer than `MAX_EMBEDDING_INPUT_LENGTH`:

1. The proxy splits the text into smaller chunks (with 10% overlap for context preservation)
2. Each chunk is sent as a separate request to Ollama sequentially
3. The proxy collects all embedding vectors
4. Embeddings are averaged to create a single combined embedding
5. The client receives one response, transparently

**Example:**

```bash
# Allow larger inputs before chunking (4000 characters)
MAX_EMBEDDING_INPUT_LENGTH=4000 cargo run --release

# Disable chunking (return error for large inputs)
ENABLE_AUTO_CHUNKING=false cargo run --release
```

**Performance Considerations:**

- Chunking processes sequentially to avoid memory pressure
- A 10,000 character input with 2000 char limit creates ~5 chunks
- Each chunk adds ~200-500ms latency (model dependent)
- For best performance, keep inputs under the limit when possible

## Flash Attention

### What is Flash Attention?

Flash Attention is an optimization technique that speeds up inference and reduces memory usage. Ollama can enable it automatically for supported models.

### How to Control Flash Attention

Flash Attention is **global only** (environment variable), not per-request:

```bash
# Let Ollama decide (RECOMMENDED - unset the variable)
unset OLLAMA_FLASH_ATTENTION
ollama serve

# Explicitly enable (may cause issues with large contexts)
export OLLAMA_FLASH_ATTENTION=1
ollama serve

# Explicitly disable (may help with large context stalls)
export OLLAMA_FLASH_ATTENTION=0
ollama serve
```

### When Flash Attention Causes Problems

**Symptoms:**

- Requests with large contexts (>60K tokens) stall indefinitely
- GPU shows "100% allocated" but 0% utilization in Activity Monitor
- Ollama process is running but not responding
- Client times out without receiving response

**Why This Happens:**
Flash attention with very large contexts can trigger memory allocation deadlocks or exceed Metal's working set limits on macOS, especially with M-series chips.

**Solutions:**

1. **Unset flash attention** (let Ollama decide per-model):

   ```bash
   unset OLLAMA_FLASH_ATTENTION
   pkill ollama
   ollama serve
   ```

2. **Reduce context size** (use the proxy's safety cap):

   ```bash
   MAX_CONTEXT_OVERRIDE=16384 cargo run --release
   ```

3. **Test systematically** to find your hardware's limits:
   ```bash
   ./test_context_limits.sh gpt-oss:20b
   ```

### Best Practices

✅ **DO:**

- Keep `OLLAMA_FLASH_ATTENTION` **unset** (let Ollama auto-detect)
- Use `MAX_CONTEXT_OVERRIDE=16384` for reliability
- Test with `test_context_limits.sh` to find your system's sweet spot
- Monitor GPU utilization when testing large contexts

❌ **DON'T:**

- Set flash attention to `false` globally (disables it for all models)
- Use contexts >60K without testing first
- Assume model's claimed context limit works reliably in practice

## Troubleshooting

### 500 Internal Server Error from Ollama

**Symptoms:**

- Embeddings requests return HTTP 500
- Ollama logs show `SIGABRT: abort` or `output_reserve: reallocating output buffer`
- Error occurs with large text inputs (> 5000 characters)

**Cause:**
Ollama's embedding models crash when trying to allocate large buffers for very long inputs.

**Solutions:**

1. **Enable chunking** (should be on by default):

   ```bash
   ENABLE_AUTO_CHUNKING=true cargo run --release
   ```

2. **Reduce chunk size** if still seeing errors:

   ```bash
   MAX_EMBEDDING_INPUT_LENGTH=1500 cargo run --release
   ```

3. **Check Ollama logs** for details:
   ```bash
   tail -f ~/.ollama/logs/server.log
   ```

### Input Too Large Error

**Symptoms:**

- Request returns HTTP 400
- Error message: "Input too large (X characters). Maximum is Y characters."

**Cause:**
Input exceeds `MAX_EMBEDDING_INPUT_LENGTH` and chunking is disabled.

**Solution:**
Enable chunking:

```bash
ENABLE_AUTO_CHUNKING=true cargo run --release
```

### Slow Embeddings Requests

**Symptoms:**

- Embeddings take much longer than expected
- Logs show "Processing X chunks sequentially"

**Cause:**
Large inputs are being chunked and processed sequentially.

**This is expected behavior!** Chunking prevents crashes but adds latency.

**To improve speed:**

1. Reduce input size at the source
2. Increase `MAX_EMBEDDING_INPUT_LENGTH` if your hardware can handle it
3. Use a smaller/faster embeddings model

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
