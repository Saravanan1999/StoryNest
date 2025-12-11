"""
Simple offline evaluation runner for StoryNest.

Run:
    python run_eval.py

This will:
- Execute a small suite of example prompts through the StoryNest pipeline
- Capture judge scores and feedback
- Write results to evals/storynest_results.jsonl
"""

import logging
from pathlib import Path

from storynest import StoryNestConfig, run_offline_eval, write_eval_results_jsonl


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    prompts = [
        "A gentle bedtime story about a kid who is nervous about starting school.",
        "A magical adventure in a friendly forest with talking animals.",
        "A short story about learning to share toys with friends.",
    ]

    config = StoryNestConfig(min_score=8.0, max_iterations=1)
    logging.info("Starting offline eval run for %d prompts", len(prompts))

    results = run_offline_eval(prompts, config=config)

    output_path = Path("evals/storynest_results.jsonl")
    write_eval_results_jsonl(results, output_path)

    logging.info("Offline eval completed. Results written to %s", output_path)
    print(f"Offline eval completed. Results written to {output_path}")


if __name__ == "__main__":
    main()


