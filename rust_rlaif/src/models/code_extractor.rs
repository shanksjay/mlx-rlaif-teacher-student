use regex::Regex;

/// Extract the first complete code block from generated text.
/// 
/// This prevents repetitive output by stopping at the first complete code block.
/// Handles patterns like:
/// - ```language\n...code...\n```
/// - Detects repetition patterns like "``` ```cpp" or duplicate code blocks
/// - Also handles code without markdown fences
pub fn extract_first_code_block(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    
    // Pattern to match code blocks: ```language\n...code...\n```
    // Also handles cases where there might be whitespace
    let code_block_pattern = Regex::new(r"```\s*(\w+)?\s*\n(.*?)\n\s*```").unwrap();
    
    if let Some(first_match) = code_block_pattern.find(text) {
        let end_pos = first_match.end();
        
        // Extract just the code content (without markdown fences)
        if let Some(caps) = code_block_pattern.captures(text) {
            if let Some(code_content) = caps.get(2) {
                let code = code_content.as_str().trim();
                if !code.is_empty() {
                    // Check if there's immediate repetition (like "``` ```cpp" right after)
                    let remaining = text[end_pos..].trim();
                    
                    // Pattern to detect repetition: closing ``` followed immediately by opening ```
                    let repetition_pattern = Regex::new(r"```\s*```").unwrap();
                    if repetition_pattern.find(&remaining[..remaining.len().min(50)]).is_some() {
                        // This is repetition, return only first block
                        return code.to_string();
                    }
                    
                    // If there are multiple blocks and the second starts very close, it's likely repetition
                    if let Some(second_match) = code_block_pattern.find(&text[end_pos..]) {
                        let gap = second_match.start();
                        // If gap is very small (< 20 chars), it's likely repetition
                        if gap < 20 {
                            return code.to_string();
                        }
                    }
                    
                    // Otherwise, return the first block code
                    return code.to_string();
                }
            }
        }
        
        // Fallback: extract with markdown if capture failed
        let first_block = text[..end_pos].trim();
        return first_block.to_string();
    }
    
    // If no markdown blocks found, extract code directly
    // Look for code patterns and collect until we hit non-code or repetition
    let lines: Vec<&str> = text.lines().collect();
    let mut result_lines: Vec<&str> = Vec::new();
    let mut in_code = false;
    let mut empty_line_count = 0;
    
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        
        // Detect code-like patterns
        if trimmed.starts_with("def ") || 
           trimmed.starts_with("class ") ||
           trimmed.starts_with("import ") ||
           trimmed.starts_with("from ") ||
           trimmed.starts_with("#include") ||
           trimmed.starts_with("fn ") ||
           trimmed.starts_with("pub fn ") ||
           trimmed.starts_with("use ") ||
           trimmed.starts_with("pub use ") ||
           trimmed.starts_with("struct ") ||
           trimmed.starts_with("impl ") ||
           trimmed.starts_with("mod ") ||
           (trimmed.starts_with("#") && idx == 0) || // Python/Rust comment at start
           (trimmed.ends_with("{") || trimmed.ends_with("}")) || // Rust/C++ braces
           (trimmed.contains("->") && trimmed.contains("fn")) { // Rust function signature
            in_code = true;
            empty_line_count = 0;
        }
        
        if in_code {
            // Stop at obvious non-code patterns or excessive empty lines
            if trimmed.starts_with("```") {
                break;
            }
            
            if trimmed.is_empty() {
                empty_line_count += 1;
                // Allow up to 2 consecutive empty lines, then stop
                if empty_line_count > 2 && result_lines.len() > 10 {
                    break;
                }
            } else {
                empty_line_count = 0;
            }
            
            result_lines.push(line);
        }
    }
    
    if !result_lines.is_empty() {
        let code = result_lines.join("\n").trim().to_string();
        // Only return if we have substantial code (at least 20 chars)
        if code.len() >= 20 {
            return code;
        }
    }
    
    // Fallback: return original text (might be pure code without patterns)
    text.trim().to_string()
}

/// Extract code from text that may include the prompt.
/// Removes the prompt prefix and extracts just the generated code.
pub fn extract_generated_code(full_text: &str, prompt: &str) -> String {
    // First, try to find where the prompt ends
    if full_text.starts_with(prompt) {
        let generated = &full_text[prompt.len()..];
        return generated.trim().to_string();
    }
    
    // If prompt not found exactly, try to find it as a substring
    // (MLX might add some whitespace or formatting)
    if let Some(pos) = full_text.find(prompt) {
        let generated = &full_text[pos + prompt.len()..];
        return generated.trim().to_string();
    }
    
    // If prompt not found, try to find common separators that indicate code start
    let separators = ["\nCode:", "\n```", "\ndef ", "\nclass ", "\nfn ", "\npub fn "];
    for sep in &separators {
        if let Some(pos) = full_text.find(sep) {
            let after_sep = &full_text[pos + 1..]; // +1 to skip the newline
            return after_sep.trim().to_string();
        }
    }
    
    // Fallback: return full text (might be pure code without prompt)
    full_text.trim().to_string()
}

/// Extract code from thinking output that may include reasoning before code.
/// Looks for code blocks (```language ... ```) or code after thinking/reasoning text.
/// 
/// Note: Currently not used but kept for future support of thinking prompts.
#[allow(dead_code)]
pub fn extract_code_from_thinking(text: &str, language: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    
    // Try to find code block first (with closing ```)
    let code_block_pattern = format!(r"```{}?\s*\n(.*?)```", regex::escape(language));
    if let Ok(re) = Regex::new(&code_block_pattern) {
        if let Some(caps) = re.captures(text) {
            if let Some(code) = caps.get(1) {
                return code.as_str().trim().to_string();
            }
        }
    }
    
    // Try generic code block (with closing ```)
    let generic_pattern = Regex::new(r"```\s*\n(.*?)```").unwrap();
    if let Some(caps) = generic_pattern.captures(text) {
        if let Some(code) = caps.get(1) {
            return code.as_str().trim().to_string();
        }
    }
    
    // Try to find code block that starts with ```{language} but doesn't close
    let escaped_lang = regex::escape(language);
    let code_block_start_pattern = format!(r"```{}?\s*\n(.*)", escaped_lang);
    if let Ok(re) = Regex::new(&code_block_start_pattern) {
        if let Some(caps) = re.captures(text) {
            if let Some(code) = caps.get(1) {
                let code = code.as_str().trim();
                // Remove any trailing ``` if present
                let code = code.trim_end_matches("```").trim();
                if !code.is_empty() {
                    return code.to_string();
                }
            }
        }
    }
    
    // Try generic code block start (without closing)
    let generic_start_pattern = Regex::new(r"```\s*\n(.*)").unwrap();
    if let Some(caps) = generic_start_pattern.captures(text) {
        if let Some(code) = caps.get(1) {
            let code = code.as_str().trim();
            // Remove any trailing ``` if present
            let code = code.trim_end_matches("```").trim();
            if !code.is_empty() {
                return code.to_string();
            }
        }
    }
    
    // If no code block, look for code-like patterns after thinking markers
    let thinking_markers = [
        "Now write the code:",
        "Implementation:",
        "Code:",
        "Solution:",
        "Here's the code:",
    ];
    
    for marker in &thinking_markers {
        if let Some(pos) = text.find(marker) {
            let after_marker = &text[pos + marker.len()..];
            let code = after_marker.trim();
            if !code.is_empty() {
                return code.to_string();
            }
        }
    }
    
    text.trim().to_string()
}

