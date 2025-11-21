#!/bin/bash
# Test script for Ollama Proxy
# This simulates the problematic requests that Elephas sends

set -e

PROXY_URL="${PROXY_URL:-http://127.0.0.1:11435}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

echo "🧪 Testing Ollama Proxy"
echo "======================="
echo "Proxy URL: $PROXY_URL"
echo "Ollama URL: $OLLAMA_URL"
echo ""

# Check if Ollama is running
echo "📡 Checking if Ollama is running..."
if ! curl -s "${OLLAMA_URL}/api/version" > /dev/null; then
    echo "❌ Error: Ollama is not running on ${OLLAMA_URL}"
    echo "Please start Ollama first."
    exit 1
fi
echo "✅ Ollama is running"
echo ""

# Check if proxy is running
echo "📡 Checking if proxy is running..."
if ! curl -s "${PROXY_URL}/api/version" > /dev/null; then
    echo "❌ Error: Proxy is not running on ${PROXY_URL}"
    echo "Please start the proxy first: cargo run"
    exit 1
fi
echo "✅ Proxy is running"
echo ""

# Get list of models
echo "📋 Fetching available models..."
MODELS=$(curl -s "${PROXY_URL}/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
if [ -z "$MODELS" ]; then
    echo "⚠️  No models found. Please pull a model first:"
    echo "  ollama pull nomic-embed-text"
    exit 1
fi
echo "Available models:"
echo "$MODELS" | head -5
echo ""

# Test 1: Embedding request with excessive num_ctx (simulating Elephas behavior)
echo "🧪 Test 1: Embedding request with num_ctx=131072 (should be reduced to 8192)"
echo "------------------------------------------------------------------------"

# Check if nomic-embed-text is available
if echo "$MODELS" | grep -q "nomic-embed-text"; then
    MODEL="nomic-embed-text"
elif echo "$MODELS" | grep -q "embed"; then
    MODEL=$(echo "$MODELS" | grep "embed" | head -1)
else
    echo "⚠️  No embedding model found. Skipping embedding test."
    MODEL=""
fi

if [ -n "$MODEL" ]; then
    echo "Using model: $MODEL"
    
    # Send request through proxy
    RESPONSE=$(curl -s -X POST "${PROXY_URL}/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"input\": \"Hello, this is a test.\",
            \"options\": {
                \"num_ctx\": 131072
            }
        }")
    
    if echo "$RESPONSE" | grep -q "embedding"; then
        echo "✅ Request successful - check proxy logs for parameter modifications"
    else
        echo "❌ Request failed:"
        echo "$RESPONSE"
    fi
else
    echo "⚠️  Skipping Test 1 - no embedding model available"
fi
echo ""

# Test 2: Chat request (should pass through unchanged if within limits)
echo "🧪 Test 2: Chat request with reasonable num_ctx"
echo "------------------------------------------------"

CHAT_MODEL=$(echo "$MODELS" | grep -v "embed" | head -1)
if [ -n "$CHAT_MODEL" ]; then
    echo "Using model: $CHAT_MODEL"
    
    RESPONSE=$(curl -s -X POST "${PROXY_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${CHAT_MODEL}\",
            \"prompt\": \"Say 'test successful'\",
            \"stream\": false,
            \"options\": {
                \"num_ctx\": 4096
            }
        }")
    
    if echo "$RESPONSE" | grep -q "response"; then
        echo "✅ Request successful - check proxy logs"
    else
        echo "❌ Request failed:"
        echo "$RESPONSE"
    fi
else
    echo "⚠️  Skipping Test 2 - no chat model available"
fi
echo ""

# Test 3: Request without num_ctx (proxy should add it for embedding models)
echo "🧪 Test 3: Embedding request without num_ctx (proxy should add it)"
echo "-------------------------------------------------------------------"

if [ -n "$MODEL" ]; then
    echo "Using model: $MODEL"
    
    RESPONSE=$(curl -s -X POST "${PROXY_URL}/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"input\": \"Test without explicit context length.\"
        }")
    
    if echo "$RESPONSE" | grep -q "embedding"; then
        echo "✅ Request successful - proxy should have added num_ctx automatically"
    else
        echo "❌ Request failed:"
        echo "$RESPONSE"
    fi
else
    echo "⚠️  Skipping Test 3 - no embedding model available"
fi
echo ""

echo "🎉 Testing complete!"
echo "Check the proxy logs to verify parameter modifications."

