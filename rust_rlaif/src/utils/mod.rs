use anyhow::Result;
use crate::models::CodeDataset;
use std::path::Path;

pub fn load_dataset<P: AsRef<Path>>(file_path: P) -> Result<CodeDataset> {
    CodeDataset::load(file_path)
}



