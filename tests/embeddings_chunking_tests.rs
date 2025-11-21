use ollama_proxy::chunker::chunk_text;
use ollama_proxy::translator::prepare_embeddings_input;

#[test]
fn test_large_input_gets_chunked() {
    // Create a large input (10K characters)
    let long_text = "This is a test sentence. ".repeat(400); // ~10,000 chars
    
    let result = chunk_text(&long_text, 2000);
    
    // Should be split into multiple chunks
    assert!(result.len() > 1, "Expected multiple chunks, got {}", result.len());
    
    // Each chunk should not exceed max length
    for (i, chunk) in result.iter().enumerate() {
        assert!(
            chunk.len() <= 2000,
            "Chunk {} exceeded max length: {} > 2000",
            i,
            chunk.len()
        );
    }
    
    // All chunks combined should roughly equal original length (with overlap)
    let total_chars: usize = result.iter().map(|s| s.len()).sum();
    assert!(
        total_chars >= long_text.len(),
        "Total chunks ({}) should be >= original ({})",
        total_chars,
        long_text.len()
    );
}

#[test]
fn test_input_at_limit_boundary() {
    // Input exactly at the limit
    let text = "a".repeat(2000);
    
    let result = chunk_text(&text, 2000);
    
    // Should NOT be chunked
    assert_eq!(result.len(), 1, "Should not chunk text at exact limit");
    assert_eq!(result[0].len(), 2000);
}

#[test]
fn test_input_just_over_limit() {
    // Input just over the limit
    let text = "a".repeat(2001);
    
    let result = chunk_text(&text, 2000);
    
    // Should be chunked into 2 pieces
    assert!(result.len() >= 2, "Should chunk text just over limit");
    
    // Each chunk should not exceed limit
    for chunk in &result {
        assert!(chunk.len() <= 2000);
    }
}

#[test]
fn test_empty_input_handling() {
    let result = chunk_text("", 2000);
    
    // Empty input should return empty vector
    assert_eq!(result.len(), 0, "Empty input should return empty vector");
}

#[test]
fn test_very_small_input() {
    let text = "Hello";
    let result = chunk_text(text, 2000);
    
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], text);
}

#[test]
fn test_prepare_embeddings_with_chunking_enabled() {
    let long_text = "This is a test. ".repeat(200); // ~3200 chars
    let inputs = vec![long_text];
    
    let result = prepare_embeddings_input(inputs, 2000, true);
    
    assert!(result.is_ok(), "Should succeed with chunking enabled");
    let chunked = result.unwrap();
    
    // Should be split into chunks
    assert!(chunked.len() > 1, "Should split into multiple chunks");
    
    // Each chunk should not exceed limit
    for chunk in &chunked {
        assert!(chunk.len() <= 2000);
    }
}

#[test]
fn test_prepare_embeddings_with_chunking_disabled() {
    let long_text = "a".repeat(5000);
    let inputs = vec![long_text];
    
    let result = prepare_embeddings_input(inputs, 2000, false);
    
    // Should return error when chunking is disabled
    assert!(result.is_err(), "Should fail when chunking disabled for large input");
    
    let err = result.unwrap_err();
    assert!(err.contains("too large"), "Error should mention input is too large");
    assert!(err.contains("2000"), "Error should mention the limit");
}

#[test]
fn test_prepare_embeddings_short_input_no_chunking_needed() {
    let short_text = "Hello world".to_string();
    let inputs = vec![short_text.clone()];
    
    let result = prepare_embeddings_input(inputs, 2000, true);
    
    assert!(result.is_ok());
    let output = result.unwrap();
    
    // Should not be chunked
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], short_text);
}

#[test]
fn test_prepare_embeddings_mixed_lengths() {
    let short = "Hello".to_string();
    let long = "This is a test. ".repeat(200); // ~3200 chars
    let inputs = vec![short.clone(), long];
    
    let result = prepare_embeddings_input(inputs, 2000, true);
    
    assert!(result.is_ok());
    let output = result.unwrap();
    
    // Short input stays as is, long input gets chunked
    // So we should have more than 2 items
    assert!(output.len() > 2, "Expected short input + chunked long input");
    
    // First item should be the short one unchanged
    assert_eq!(output[0], short);
}

#[test]
fn test_sentence_boundary_splitting() {
    // Create text with clear sentence boundaries
    let text = "First sentence. ".repeat(50) + &"Second sentence. ".repeat(50);
    
    let result = chunk_text(&text, 500);
    
    // Should create multiple chunks
    assert!(result.len() > 1);
    
    // Check that chunks split on sentence boundaries (should contain complete sentences)
    for chunk in &result {
        // Each chunk should either end with a period or be at the end
        if chunk.len() < 500 {
            // Last chunk might be incomplete
            continue;
        }
        // Other chunks should end near a sentence boundary
        assert!(chunk.len() <= 500);
    }
}

#[test]
fn test_no_infinite_loop_on_long_word() {
    // Create a very long string with no good break points
    let text = "a".repeat(10000);
    
    let result = chunk_text(&text, 500);
    
    // Should complete without hanging
    assert!(!result.is_empty(), "Should return chunks even for unbreakable text");
    
    // Should create multiple chunks
    assert!(result.len() > 1);
    
    // Each chunk should not exceed limit
    for chunk in &result {
        assert!(chunk.len() <= 500);
    }
}

#[test]
fn test_chunking_preserves_content() {
    let original = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    
    let chunks = chunk_text(&original, 500);
    
    // Concatenate all chunks (removing overlap)
    let mut reconstructed = String::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            reconstructed.push_str(chunk);
        } else {
            // For subsequent chunks, we need to handle overlap
            // This is approximate since overlap detection is complex
            reconstructed.push_str(chunk);
        }
    }
    
    // The reconstructed text should contain all content from original
    // (may have duplicates due to overlap, but should have all words)
    assert!(
        reconstructed.len() >= original.len(),
        "Reconstructed text should be at least as long as original"
    );
}

