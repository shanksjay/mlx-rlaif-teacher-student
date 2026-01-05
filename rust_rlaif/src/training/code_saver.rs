use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIndexEntry {
    pub prompt: String,
    pub language: String,
    pub source: String,
    pub filepath: String,
    pub absolute_path: String,
    pub timestamp: String,
    pub hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advantage: Option<f64>,
}

pub struct CodeSaver {
    output_dir: PathBuf,
    index: HashMap<String, CodeIndexEntry>,
    index_file: PathBuf,
}

impl CodeSaver {
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;
        
        let index_file = output_dir.join("index.json");
        
        // Load existing index if it exists
        let index = if index_file.exists() {
            let content = fs::read_to_string(&index_file)
                .with_context(|| format!("Failed to read index file: {:?}", index_file))?;
            serde_json::from_str(&content)
                .unwrap_or_else(|_| HashMap::new())
        } else {
            HashMap::new()
        };
        
        Ok(Self {
            output_dir,
            index,
            index_file,
        })
    }
    
    fn generate_hash(prompt: &str, language: &str, source: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        language.hash(&mut hasher);
        source.hash(&mut hasher);
        let hash_val = hasher.finish();
        format!("{:016x}", hash_val)[..12].to_string()
    }
    
    pub fn save_code(
        &mut self,
        code: &str,
        prompt: &str,
        language: &str,
        source: &str,
        reward: Option<f64>,
        advantage: Option<f64>,
        scores: Option<HashMap<String, f64>>,
    ) -> Result<CodeIndexEntry> {
        // Create source-specific directory
        let source_dir = self.output_dir.join(source);
        fs::create_dir_all(&source_dir)
            .with_context(|| format!("Failed to create source directory: {:?}", source_dir))?;
        
        // Generate filename
        let hash = Self::generate_hash(prompt, language, source);
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("{}_{}.{}", timestamp, hash, language);
        let filepath = source_dir.join(&filename);
        
        // Save code
        fs::write(&filepath, code)
            .with_context(|| format!("Failed to write code file: {:?}", filepath))?;
        
        // Create index entry
        let entry = CodeIndexEntry {
            prompt: prompt.to_string(),
            language: language.to_string(),
            source: source.to_string(),
            filepath: filepath
                .strip_prefix(&self.output_dir)
                .unwrap_or(&filepath)
                .to_string_lossy()
                .to_string(),
            absolute_path: filepath.to_string_lossy().to_string(),
            timestamp,
            hash: hash.clone(),
            scores,
            reward,
            advantage,
        };
        
        // Add to index
        let index_key = format!("{}_{}", source, hash);
        self.index.insert(index_key, entry.clone());
        
        debug!("Saved generated code: {} -> {:?}", prompt.chars().take(50).collect::<String>(), filepath);
        
        Ok(entry)
    }
    
    pub fn save_index(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.index)
            .context("Failed to serialize index to JSON")?;
        fs::write(&self.index_file, content)
            .with_context(|| format!("Failed to write index file: {:?}", self.index_file))?;
        Ok(())
    }
    
    pub fn get_output_dir(&self) -> &Path {
        &self.output_dir
    }
}

