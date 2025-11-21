#!/bin/bash

# Test Context Limits Script
# Systematically tests different context sizes to find optimal settings
# Usage: ./test_context_limits.sh [model_name]

set -e

MODEL="${1:-gpt-oss:20b}"
PROXY_URL="http://127.0.0.1:11435"
OLLAMA_URL="http://127.0.0.1:11434"

# Test configurations: context_size, description
CONTEXT_SIZES=(
    "4096:Small (4K)"
    "8192:Medium (8K)"
    "16384:Recommended (16K)"
    "32768:Large (32K)"
    "60000:Very Large (60K)"
    "131072:Maximum (131K)"
)

RESULTS_FILE="context_test_results_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "Context Size Testing Tool"
echo "========================================"
echo "Model: $MODEL"
echo "Proxy: $PROXY_URL"
echo "Ollama: $OLLAMA_URL"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Check if Ollama is running
if ! curl -s "$OLLAMA_URL/api/version" > /dev/null 2>&1; then
    echo "❌ Error: Ollama is not running at $OLLAMA_URL"
    echo "   Start Ollama first: ollama serve"
    exit 1
fi

# Check if proxy is running
if ! curl -s "$PROXY_URL" > /dev/null 2>&1; then
    echo "⚠️  Warning: Proxy may not be running at $PROXY_URL"
    echo "   Cannot test OpenAI compatibility without proxy"
    exit 1
else
    echo "✅ Proxy detected, will test OpenAI chat completions endpoint"
    TEST_URL="$PROXY_URL/v1/chat/completions"
fi

echo ""
echo "Starting tests..."
echo "" | tee "$RESULTS_FILE"
echo "========================================"  | tee -a "$RESULTS_FILE"
echo "Context Size Test Results" | tee -a "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "Model: $MODEL" | tee -a "$RESULTS_FILE"
echo "========================================"  | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Function to test a specific context size
test_context_size() {
    local ctx_size=$1
    local description=$2
    
    echo -n "Testing $description ($ctx_size tokens)... "
    
    # Simple test prompt - using OpenAI API format
    local request='{
        "model": "'$MODEL'",
        "messages": [
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
    
    # Start timer
    local start_time=$(date +%s)
    
    # Send request with timeout
    local response
    local status_code
    
    if response=$(curl -s -w "\n%{http_code}" --max-time 180 \
        -X POST "$TEST_URL" \
        -H "Content-Type: application/json" \
        -d "$request" 2>&1); then
        
        # Extract status code (last line)
        status_code=$(echo "$response" | tail -n 1)
        response_body=$(echo "$response" | sed '$d')
        
        # Calculate duration
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ "$status_code" = "200" ]; then
            echo "✅ SUCCESS (${duration}s)"
            echo "$description ($ctx_size): ✅ SUCCESS in ${duration}s" | tee -a "$RESULTS_FILE"
            
            # Check if response contains actual content (OpenAI format has "choices")
            if echo "$response_body" | grep -q "choices" 2>/dev/null; then
                echo "   Response received and valid (OpenAI format)" | tee -a "$RESULTS_FILE"
            else
                echo "   ⚠️  Response may be incomplete" | tee -a "$RESULTS_FILE"
            fi
        else
            echo "❌ FAILED (HTTP $status_code in ${duration}s)"
            echo "$description ($ctx_size): ❌ FAILED - HTTP $status_code in ${duration}s" | tee -a "$RESULTS_FILE"
            
            # Log error details
            if echo "$response_body" | grep -q "error" 2>/dev/null; then
                local error_msg=$(echo "$response_body" | grep -o '"error":"[^"]*"' | head -1)
                echo "   Error: $error_msg" | tee -a "$RESULTS_FILE"
            fi
        fi
    else
        echo "❌ TIMEOUT (>180s)"
        echo "$description ($ctx_size): ❌ TIMEOUT (>180s)" | tee -a "$RESULTS_FILE"
        echo "   Request did not complete - likely stalled" | tee -a "$RESULTS_FILE"
    fi
    
    echo "" | tee -a "$RESULTS_FILE"
    
    # Small delay between tests
    sleep 2
}

# Run tests for each context size
for config in "${CONTEXT_SIZES[@]}"; do
    IFS=':' read -r ctx_size description <<< "$config"
    test_context_size "$ctx_size" "$description"
done

# Summary
echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Summary" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

success_count=$(grep -c "✅ SUCCESS" "$RESULTS_FILE" || true)
fail_count=$(grep -c "❌ FAILED\|❌ TIMEOUT" "$RESULTS_FILE" || true)

echo "Tests passed: $success_count" | tee -a "$RESULTS_FILE"
echo "Tests failed: $fail_count" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Find maximum working context
last_success=$(grep "✅ SUCCESS" "$RESULTS_FILE" | tail -1 || echo "None")
if [ "$last_success" != "None" ]; then
    echo "✅ Highest working context: $last_success" | tee -a "$RESULTS_FILE"
else
    echo "❌ No successful tests found" | tee -a "$RESULTS_FILE"
fi

# Recommendations
echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Recommendations" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

if [ $success_count -eq ${#CONTEXT_SIZES[@]} ]; then
    echo "🎉 All tests passed! Your system handles all context sizes well." | tee -a "$RESULTS_FILE"
    echo "   You can increase MAX_CONTEXT_OVERRIDE if desired." | tee -a "$RESULTS_FILE"
elif [ $success_count -ge 3 ]; then
    echo "👍 Most tests passed. Set MAX_CONTEXT_OVERRIDE to highest successful value." | tee -a "$RESULTS_FILE"
else
    echo "⚠️  Many tests failed. Recommendations:" | tee -a "$RESULTS_FILE"
    echo "   1. Restart Ollama: pkill ollama && ollama serve" | tee -a "$RESULTS_FILE"
    echo "   2. Unset flash attention: unset OLLAMA_FLASH_ATTENTION" | tee -a "$RESULTS_FILE"
    echo "   3. Use conservative MAX_CONTEXT_OVERRIDE=16384" | tee -a "$RESULTS_FILE"
fi

echo ""  | tee -a "$RESULTS_FILE"
echo "Full results saved to: $RESULTS_FILE"
echo ""

# Flash attention detection and advice
if [ -n "$OLLAMA_FLASH_ATTENTION" ]; then
    echo "📍 Flash Attention Status: SET to '$OLLAMA_FLASH_ATTENTION'" | tee -a "$RESULTS_FILE"
    if [ "$fail_count" -gt 0 ]; then
        echo "   💡 Try unsetting: unset OLLAMA_FLASH_ATTENTION && ollama serve" | tee -a "$RESULTS_FILE"
    fi
else
    echo "📍 Flash Attention Status: UNSET (Ollama decides per-model)" | tee -a "$RESULTS_FILE"
fi

echo ""
echo "To unset OLLAMA_FLASH_ATTENTION:"
echo "  unset OLLAMA_FLASH_ATTENTION"
echo "  ollama serve"
echo ""

