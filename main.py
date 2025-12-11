"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

Answer:
Right now, the project is a single-turn story generator. There are several improvements that could be made to the project.
- Add a multi-turn conversation support to support follow-up questions and conversations.
- Extend the offline eval tooling to support multiple configs/prompt variants (A/B experiments), tagging each EvalResult with an experiment_name and adding a simple aggregation script/notebook to compare average safety/final scores across experiments.
- Add a pluggable LLM backend interface (real OpenAI vs. deterministic fake) so the same StoryNest pipeline can be run in CI without network calls, making it easy to have fast, fully offline tests that validate orchestration, logging, and metrics behavior.
- Wrap generate_bedtime_story in a minimal FastAPI/Flask endpoints to allow for easy deployment and integration with other applications.
- Add an adversarial ‘jailbreak’ stress-test suite of dark/edge-case prompts to systematically probe and harden the SafetyGuard and Judge prompts.

"""

import logging

from storynest import generate_bedtime_story
from storynest.llm_client import LLMClientError


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    user_input = input("What kind of story do you want to hear? ")

    try:
        response = generate_bedtime_story(user_input)
        print(response)
    except LLMClientError as exc:
        logging.error("Could not generate story due to LLM error: %s", exc)
        print("Sorry, there was a problem talking to the story engine. Please try again later.")


if __name__ == "__main__":
    main()