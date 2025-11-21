# Implementation Summary

## Problem Solved

Elephas uses Ollama's OpenAI-compatible API (`/v1/embeddings`), which does not accept runtime `options` parameters. This causes all requests to use the global `OLLAMA_CONTEXT_LENGTH` setting (131072), even for embedding models trained with 8192 tokens.

## Solution: API Translation Proxy

The proxy translates between API formats:

### Request Flow

1. **Receive OpenAI Request**

   ```json
   POST /v1/embeddings
   {"model": "snowflake-arctic-embed2", "input": ["text"]}
   ```

2. **Fetch Model Metadata**

   - Query `/api/show` for model's `n_ctx_train`
   - Cache result for performance

3. **Translate to Ollama Native API**

   ```json
   POST /api/embed
   {
     "model": "snowflake-arctic-embed2",
     "input": ["text"],
     "options": {"num_ctx": 8192},
     "truncate": true
   }
   ```

4. **Ollama Processes with Correct Context**

   - Uses `num_ctx: 8192` from request
   - Ignores global `OLLAMA_CONTEXT_LENGTH`

5. **Translate Response Back**
   ```json
   Ollama: {"embeddings": [[...]]}
   →
   OpenAI: {"object": "list", "data": [{"embedding": [...]}]}
   ```

## Implementation Details

### Key Files

- **`src/translator.rs`** - API format conversion

  - Request translation: OpenAI → Ollama
  - Response translation: Ollama → OpenAI
  - Endpoint mapping

- **`src/proxy.rs`** - Request routing

  - Detects OpenAI endpoints
  - Routes to translation handler
  - Handles standard pass-through

- **`src/model_metadata.rs`** - Model info caching
  - Fetches `n_ctx_train` from Ollama
  - Caches per model

### Why This Works

OpenAI-compatible endpoints (`/v1/*`) in Ollama:

- ❌ Ignore runtime `options` parameters
- ✅ Only respect global env vars

Native Ollama endpoints (`/api/*`):

- ✅ Accept per-request `options`
- ✅ Override global settings

By translating between formats, we get the best of both:

- Elephas continues using OpenAI API (no config change)
- Proxy controls `num_ctx` per request (via native API)
- Each model gets appropriate context length

## Benefits

1. **No client changes** - Elephas works as-is
2. **No global setting changes** - Keep 131072 for chat models
3. **Per-model control** - Each model uses its training context
4. **Extensible** - Framework supports future translations

## Verification

Run proxy and check logs:

```
📨 Incoming request: POST /v1/embeddings
🔍 Detected model: snowflake-arctic-embed2:latest
📊 Model metadata - n_ctx_train: 8192
🔄 Translating OpenAI request to Ollama native API
✏️  Added options.num_ctx: 8192
📤 Translated request: {...}
✅ Translated response back to OpenAI format
```

Then verify with `ollama ps` - context should show 8192, not 131072.
