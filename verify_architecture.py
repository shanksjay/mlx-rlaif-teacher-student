import sys
import os
import json

# Ensure we can import rlaif-client from its source
sys.path.append(os.path.join(os.getcwd(), "rlaif-client"))

import rlaif_core
from rlaif_client.backends import TorchBackend
# rlaif_core now has the TrainingLoop class

def main():
    print("=== RLAIF Architecture Verification ===")

    # 1. Verify Rust Core System Capabilities
    print("\n[1] Checking System Capabilities (Rust Core)...")
    specs = rlaif_core.detect_system_capabilities()
    print(f"    {specs}")

    # 2. Verify Config Validation
    print("\n[2] Validating Config (Rust Core)...")
    valid_agent = rlaif_core.validate_agent_config('{"task_intent": "scheduler", "agent_behavior": "helpful", "constraints": []}')
    print(f"    Agent Config Valid: {valid_agent}")

    # 3. Verify Rust-Driven Training Loop
    print("\n[3] Running Rust-driven Training Loop with Python Backend...")
    backend = TorchBackend("Qwen/Qwen2.5-Coder-3B-Instruct")

    # Instantiate the Rust struct
    loop = rlaif_core.TrainingLoop()

    # The run_step method calls back into Python
    # run_step(backend, prompts, config)
    prompts = ["Write a fibonacci function", "Write a binary search"]
    result_json = loop.run_step(backend, prompts, "{}")

    print(f"    Rust Loop Result: {result_json}")

    # Verify the result is valid JSON
    result = json.loads(result_json)
    if result["samples_count"] == 2:
        print("    SUCCESS: Loop processed 2 samples.")
    else:
        print("    FAILURE: Sample count mismatch.")

if __name__ == "__main__":
    main()
