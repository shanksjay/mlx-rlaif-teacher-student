use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSample {
    pub prompt: String,
    pub language: String,
    pub code: Option<String>,
}

#[derive(Debug)]
pub struct CodeDataset {
    samples: Vec<CodeSample>,
}

impl CodeDataset {
    pub fn load<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file = File::open(&file_path)
            .with_context(|| format!("Failed to open dataset file: {:?}", file_path.as_ref()))?;
        
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.context("Failed to read line")?;
            if line.trim().is_empty() {
                continue;
            }

            let sample: CodeSample = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse JSON on line {}", line_num + 1))?;
            
            samples.push(sample);
        }

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn get(&self, index: usize) -> Option<&CodeSample> {
        self.samples.get(index)
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &CodeSample> {
        self.samples.iter()
    }
}

