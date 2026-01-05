use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TeacherModel {
    provider: String,
    model: String,
    api_key: String,
    client: Client,
    score_cache: Arc<DashMap<String, (f64, f64)>>, // (score, timestamp)
    gen_cache: Arc<DashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

impl TeacherModel {
    pub fn new(provider: &str, model: &str, api_key_env: &str) -> Result<Self> {
        let api_key = env::var(api_key_env)
            .with_context(|| format!("Failed to read API key from {}", api_key_env))?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;

        Ok(Self {
            provider: provider.to_string(),
            model: model.to_string(),
            api_key,
            client,
            score_cache: Arc::new(DashMap::new()),
            gen_cache: Arc::new(DashMap::new()),
        })
    }

    pub async fn generate(&self, prompt: &str, language: &str) -> Result<String> {
        // Check cache
        let cache_key = format!("gen:{}:{}", prompt, language);
        if let Some(cached) = self.gen_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let formatted_prompt = format!("Write high-quality {} code:\n\n{}\n\nCode:", language, prompt);
        
        let code = match self.provider.as_str() {
            "anthropic" => self.generate_anthropic(&formatted_prompt).await?,
            "openai" => self.generate_openai(&formatted_prompt).await?,
            _ => anyhow::bail!("Unsupported provider: {}", self.provider),
        };

        // Cache result
        self.gen_cache.insert(cache_key, code.clone());
        Ok(code)
    }

    async fn generate_anthropic(&self, prompt: &str) -> Result<String> {
        let url = "https://api.anthropic.com/v1/messages";
        
        #[derive(Serialize)]
        struct AnthropicRequest {
            model: String,
            max_tokens: usize,
            messages: Vec<Message>,
        }

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 2048,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        };

        let response = self.client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct AnthropicResponse {
            content: Vec<ContentBlock>,
        }

        #[derive(Deserialize)]
        struct ContentBlock {
            text: String,
        }

        let api_response: AnthropicResponse = response.json().await?;
        
        if let Some(block) = api_response.content.first() {
            Ok(block.text.clone())
        } else {
            anyhow::bail!("No response from API")
        }
    }

    async fn generate_openai(&self, prompt: &str) -> Result<String> {
        let url = "https://api.openai.com/v1/chat/completions";
        
        #[derive(Serialize)]
        struct OpenAiRequest {
            model: String,
            messages: Vec<Message>,
            temperature: f64,
            max_tokens: usize,
        }

        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct OpenAiChoice {
            message: Message,
        }

        #[derive(Deserialize)]
        struct OpenAiResponse {
            choices: Vec<OpenAiChoice>,
        }

        let api_response: OpenAiResponse = response.json().await?;
        
        if let Some(choice) = api_response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            anyhow::bail!("No response from API")
        }
    }

    pub async fn score_code(&self, code: &str, prompt: &str, language: &str) -> Result<f64> {
        // Check cache
        let cache_key = format!("score:{}:{}:{}", code, prompt, language);
        if let Some(cached) = self.score_cache.get(&cache_key) {
            return Ok(cached.0);
        }

        let scoring_prompt = self.build_scoring_prompt(code, prompt, language);
        
        let score_text = match self.provider.as_str() {
            "anthropic" => self.generate_anthropic(&scoring_prompt).await?,
            "openai" => self.generate_openai(&scoring_prompt).await?,
            _ => anyhow::bail!("Unsupported provider: {}", self.provider),
        };

        // Parse score from response
        let score = self.parse_score(&score_text)?;

        // Cache result
        self.score_cache.insert(cache_key, (score, chrono::Utc::now().timestamp() as f64));
        Ok(score)
    }

    fn build_scoring_prompt(&self, code: &str, prompt: &str, language: &str) -> String {
        format!(
            r#"Evaluate the following {} code on a scale of 0.0 to 1.0.
For each criterion, assign a score where 1.0 is perfect, 0.5 is functional but flawed, and 0.0 is failed/missing.

1. Correctness (0.3): Does it solve the problem correctly?
2. Code Quality (0.3): Is it clean, readable, and well-structured?
3. Efficiency (0.2): Is it efficient and follows best practices?
4. Documentation (0.2): Is it well-documented?

Compute the final score as the weighted sum:
final = 0.3*correctness + 0.3*code_quality + 0.2*efficiency + 0.2*documentation

Prompt: {}

Code:
```{}
{}
```

IMPORTANT: Respond with ONLY a single float between 0.0 and 1.0 (e.g., 0.75). Do not include explanations, additional text, or newlines. Just the number."#,
            language, prompt, language, code
        )
    }

    fn parse_score(&self, text: &str) -> Result<f64> {
        // Extract first float from text
        let cleaned = text.trim().lines().next().unwrap_or("").trim();
        let score = cleaned.parse::<f64>()
            .with_context(|| format!("Failed to parse score from: {}", cleaned))?;
        
        // Clamp to [0.0, 1.0]
        Ok(score.max(0.0).min(1.0))
    }
}

