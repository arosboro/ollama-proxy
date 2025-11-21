# Quick Start Guide

## Start the Proxy

```bash
cd /Users/arosboro/git/ollama-proxy
cargo run --release
```

You should see:

```
Starting Ollama Proxy
Listening on: 127.0.0.1:11435
Proxying to: http://127.0.0.1:11434
Ollama Proxy is ready
```

## Configure Elephas

### Basic Setup

1. Open Elephas settings
2. Find the "Ollama Host" or "API Endpoint" setting
3. Change from: `http://127.0.0.1:11434`
4. Change to: `http://127.0.0.1:11435`
5. Save settings

### Elephas-Specific Configuration Guide

**Known Issues with Elephas:**

1. **Context Tokens setting may not be respected**

   - You can set "Context Tokens" to 60000 in Elephas UI
   - But Elephas often **doesn't send `num_ctx` in the API request**
   - The proxy/Ollama uses defaults or global settings instead
   - **Impact:** You're not actually getting the context size you configured

2. **Temperature setting may not be respected**

   - You set Temperature to 0.4 in Elephas
   - But request shows Temperature: 0.5
   - This is an Elephas bug, not the proxy

3. **Tone/Style settings are sent correctly**
   - "Business Marketing" tone → appears in system role ✅
   - These do work as expected

**Optimal Elephas Settings for Reliability:**

| Setting            | Recommended Value        | Why                                     |
| ------------------ | ------------------------ | --------------------------------------- |
| **Model**          | `gpt-oss:20b` or smaller | Tested and working                      |
| **Context Tokens** | Don't rely on this       | Elephas often ignores it                |
| **Temperature**    | 0.4-0.7                  | May not be respected, but set it anyway |
| **Max Tokens**     | 4096                     | This IS respected                       |
| **Tone**           | Your preference          | This works correctly                    |

**Instead of relying on Elephas settings, configure the proxy:**

```bash
# Set safe context limit (prevents stalls)
MAX_CONTEXT_OVERRIDE=16384 \
REQUEST_TIMEOUT_SECONDS=120 \
cargo run --release
```

This ensures context never exceeds 16K regardless of what Elephas sends (or doesn't send).

**For Larger Contexts (Advanced):**

If you need larger contexts and have tested your system:

```bash
# Test first with test_context_limits.sh
./test_context_limits.sh gpt-oss:20b

# If 32K works reliably, you can increase:
MAX_CONTEXT_OVERRIDE=32768 \
REQUEST_TIMEOUT_SECONDS=180 \
cargo run --release
```

**Warning Signs to Watch For:**

- ⏱️ Request times out → Reduce `MAX_CONTEXT_OVERRIDE`
- 🔴 Elephas shows "Request timed out. No context found" → Ollama stalled
- 💻 GPU at 100% allocated but 0% active → Flash attention deadlock
- 🔄 Response takes >2 minutes → Context too large for your hardware

**Quick Fixes:**

1. **Stalled requests:**

   ```bash
   # Restart Ollama without flash attention
   pkill ollama
   unset OLLAMA_FLASH_ATTENTION
   ollama serve
   ```

2. **Consistent timeouts:**

   ```bash
   # Use smaller context
   MAX_CONTEXT_OVERRIDE=8192 cargo run --release
   ```

3. **Embeddings still crashing:**
   ```bash
   # Reduce chunk size further
   MAX_EMBEDDING_INPUT_LENGTH=500 cargo run --release
   ```

## Test It

Run the test script:

```bash
./test_proxy.sh
```

Or test manually with curl:

```bash
# This simulates Elephas sending 131072 tokens to an embedding model
curl -X POST http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "test",
    "options": {
      "num_ctx": 131072
    }
  }'
```

## What to Look For

In the proxy logs, you should see:

```
📨 Incoming request: POST /v1/embeddings
🔍 Detected model: nomic-embed-text
📊 Model metadata - n_ctx_train: 8192
⚠️  num_ctx (131072) exceeds model training context (8192)
✏️  Modified options.num_ctx: 131072 → 8192
🔧 ContextLimitModifier applied modifications
📬 Response status: 200 OK
```

This means the proxy successfully:

1. Intercepted the request
2. Detected the model
3. Found it was trained with 8192 tokens
4. Reduced the requested 131072 to 8192
5. Forwarded the corrected request to Ollama

## Running in Background

To run as a background service:

```bash
# Using screen
screen -S ollama-proxy
cargo run --release
# Press Ctrl+A, then D to detach

# To reattach
screen -r ollama-proxy

# Or using nohup
nohup cargo run --release > proxy.log 2>&1 &
```

## Troubleshooting

**Proxy won't start:**

- Check if port 11435 is already in use: `lsof -i :11435`
- Try a different port: `PROXY_PORT=11436 cargo run`

**Can't connect to Ollama:**

- Verify Ollama is running: `curl http://127.0.0.1:11434/api/version`
- Check OLLAMA_HOST setting

**No modifications being logged:**

- Ensure your client is pointing to the proxy (11435), not Ollama (11434)
- Check that your model name is correct
- Try increasing log level: `RUST_LOG=debug cargo run`
