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

1. Open Elephas settings
2. Find the "Ollama Host" or "API Endpoint" setting
3. Change from: `http://127.0.0.1:11434`
4. Change to: `http://127.0.0.1:11435`
5. Save settings

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

