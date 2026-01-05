fn main() {
    // Configure PyO3 to use the correct Python executable
    // Prefer the venv Python if available, otherwise use python3
    // NOTE: cargo:rustc-env sets build-time env vars, but PyO3 also checks runtime PYO3_PYTHON
    if let Ok(python_exe) = std::env::var("PYO3_PYTHON") {
        println!("cargo:rustc-env=PYO3_PYTHON={}", python_exe);
    } else {
        // Try to find .venv/bin/python in current directory or parent directories
        let current_dir = std::env::current_dir().ok();
        let mut venv_python = None;
        
        if let Some(mut dir) = current_dir {
            // Check current directory and up to 3 levels up
            for _ in 0..4 {
                let venv_path = dir.join(".venv").join("bin").join("python");
                if venv_path.exists() {
                    if let Some(path_str) = venv_path.to_str() {
                        venv_python = Some(path_str.to_string());
                        break;
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }
        
        if let Some(venv_path) = venv_python {
            println!("cargo:rustc-env=PYO3_PYTHON={}", venv_path);
        } else {
            // Try to find python in PATH (respects venv)
            if let Ok(output) = std::process::Command::new("which")
                .arg("python")
                .output()
            {
                if let Ok(path) = String::from_utf8(output.stdout) {
                    let path = path.trim();
                    if !path.is_empty() {
                        println!("cargo:rustc-env=PYO3_PYTHON={}", path);
                    }
                }
            }
        }
    }
    
    // On macOS, PyO3 needs help finding the Python framework at runtime
    // We'll set DYLD_FRAMEWORK_PATH in the wrapper scripts, but we can also
    // try to add an rpath during linking if the framework is found
    
    #[cfg(target_os = "macos")]
    {
        // Try to find Python framework and add rpath
        let framework_paths = [
            "/Library/Developer/CommandLineTools/Library/Frameworks",
            "/Library/Frameworks",
            "/System/Library/Frameworks",
        ];
        
        for path in &framework_paths {
            let framework = std::path::Path::new(path).join("Python3.framework");
            if framework.exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
                println!("cargo:rustc-link-arg=-F{}", path);
                break;
            }
        }
    }
}

