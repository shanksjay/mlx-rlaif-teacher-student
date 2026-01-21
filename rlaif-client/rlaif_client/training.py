from typing import List, Dict, Any

class TrainingLoop:
    def __init__(self, backend, config: Dict[str, Any]):
        self.backend = backend
        self.config = config

    def run_step(self, prompts: List[str]):
        """Orchestrates a single training step."""
        print("Step 1: Generating samples...")
        samples = self.backend.generate_samples(prompts, self.config)

        print("Step 2: Scoring (Mock)...")
        scores = [0.9] * len(samples) # Mock scoring

        print("Step 3: Updating...")
        # In a real loop, we'd calculate advantage/loss here based on scores
        loss = 0.5
        self.backend.update_weights(loss)

        return {"samples": samples, "scores": scores, "loss": loss}
