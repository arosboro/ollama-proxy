/// Smart text chunking module for handling large embeddings inputs
use tracing::debug;

/// Chunk text into smaller pieces that don't exceed max_len
/// 
/// Strategy:
/// 1. Try to split on sentence boundaries (. ! ?)
/// 2. Fall back to word boundaries if sentences are too long
/// 3. Add 10% overlap between chunks for context preservation
/// 4. Ensure no chunk exceeds max_len
pub fn chunk_text(input: &str, max_len: usize) -> Vec<String> {
    // Handle empty or very short input
    if input.is_empty() {
        return vec![];
    }
    
    if input.len() <= max_len {
        return vec![input.to_string()];
    }

    debug!("Chunking text of length {} with max_len {}", input.len(), max_len);

    let mut chunks = Vec::new();
    let overlap_size = (max_len as f32 * 0.1) as usize;
    
    let mut start = 0;
    let mut prev_end = 0;
    
    while start < input.len() {
        let remaining = input.len() - start;
        
        // If remaining text fits in one chunk, take it all
        if remaining <= max_len {
            chunks.push(input[start..].to_string());
            break;
        }
        
        // Try to find a good breaking point
        let end = start + max_len;
        let chunk_end = find_break_point(&input[start..end], max_len);
        
        let actual_end = start + chunk_end;
        chunks.push(input[start..actual_end].to_string());
        
        // Ensure we make progress (avoid infinite loop)
        if actual_end <= prev_end {
            break;
        }
        prev_end = actual_end;
        
        // Move start forward, but keep overlap
        start = actual_end.saturating_sub(overlap_size);
    }
    
    debug!("Created {} chunks from input", chunks.len());
    chunks
}

/// Find the best breaking point in text, preferring sentence/word boundaries
fn find_break_point(text: &str, max_pos: usize) -> usize {
    if text.len() <= max_pos {
        return text.len();
    }
    
    // Look for sentence endings (. ! ?) in the last 20% of the chunk
    let search_start = (max_pos as f32 * 0.8) as usize;
    
    // Search backwards from max_pos for sentence boundary
    for i in (search_start..max_pos).rev() {
        if let Some(ch) = text.chars().nth(i) {
            if matches!(ch, '.' | '!' | '?') {
                // Check if there's whitespace after (proper sentence end)
                if i + 1 < text.len() {
                    if let Some(next_ch) = text.chars().nth(i + 1) {
                        if next_ch.is_whitespace() {
                            return i + 2; // Include the punctuation and space
                        }
                    }
                }
                return i + 1; // Include the punctuation
            }
        }
    }
    
    // If no sentence boundary found, look for word boundary (space)
    for i in (search_start..max_pos).rev() {
        if let Some(ch) = text.chars().nth(i) {
            if ch.is_whitespace() {
                return i + 1; // Start next chunk after the space
            }
        }
    }
    
    // If no good boundary found, split at max_pos
    max_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let result = chunk_text("", 100);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_char() {
        let result = chunk_text("a", 100);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "a");
    }

    #[test]
    fn test_short_text_no_chunking_needed() {
        let text = "Hello world";
        let result = chunk_text(text, 100);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], text);
    }

    #[test]
    fn test_exact_boundary() {
        let text = "Hello";
        let result = chunk_text(text, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Hello");
    }

    #[test]
    fn test_sentence_boundary_split() {
        let text = "First sentence. Second sentence. Third sentence.";
        let result = chunk_text(text, 30);
        
        // Should split on sentence boundaries
        assert!(result.len() >= 2);
        
        // Each chunk should not exceed max length
        for chunk in &result {
            assert!(chunk.len() <= 30, "Chunk exceeded max length: {}", chunk.len());
        }
    }

    #[test]
    fn test_word_boundary_fallback() {
        let text = "This is a very long sentence without proper punctuation that should split on word boundaries";
        let result = chunk_text(text, 40);
        
        assert!(result.len() >= 2);
        
        // Each chunk should not exceed max length
        for chunk in &result {
            assert!(chunk.len() <= 40);
        }
        
        // Should not split mid-word
        for chunk in &result {
            assert!(!chunk.ends_with(" "));
        }
    }

    #[test]
    fn test_overlap_preservation() {
        let text = "First part. Second part. Third part. Fourth part.";
        let result = chunk_text(text, 25);
        
        // With overlap, we should have some text repeated between chunks
        if result.len() > 1 {
            // Check that there's some overlap (at least one word)
            let last_words_of_first: Vec<&str> = result[0].split_whitespace().collect();
            let first_words_of_second: Vec<&str> = result[1].split_whitespace().collect();
            
            if !last_words_of_first.is_empty() && !first_words_of_second.is_empty() {
                // There should be some word overlap
                let overlap_found = last_words_of_first.iter().any(|word| {
                    first_words_of_second.contains(word)
                });
                assert!(overlap_found, "Expected overlap between chunks");
            }
        }
    }

    #[test]
    fn test_very_long_text() {
        let text = "a".repeat(10000);
        let result = chunk_text(&text, 500);
        
        // Should create multiple chunks
        assert!(result.len() > 1);
        
        // Each chunk should not exceed max length
        for chunk in &result {
            assert!(chunk.len() <= 500);
        }
        
        // Total length should roughly equal original (minus some overlap)
        let total_chars: usize = result.iter().map(|s| s.len()).sum();
        assert!(total_chars >= text.len());
    }

    #[test]
    fn test_find_break_point_sentence() {
        let text = "Hello world. This is a test.";
        let break_point = find_break_point(text, 20);
        
        // Should break on a sentence boundary
        assert!(break_point <= 20);
        // The break should be at a sentence ending
        let chunk = &text[..break_point];
        assert!(chunk.contains('.'), "Should break on sentence boundary");
    }

    #[test]
    fn test_find_break_point_word() {
        let text = "Hello world this is a test";
        let break_point = find_break_point(text, 15);
        
        // Should break on word boundary
        assert!(break_point <= 15);
        assert!(!text[..break_point].ends_with(" "));
    }

    #[test]
    fn test_no_infinite_loop() {
        // Test with text that has no good break points
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(100);
        let result = chunk_text(&text, 50);
        
        // Should complete without hanging
        assert!(!result.is_empty());
    }
}

