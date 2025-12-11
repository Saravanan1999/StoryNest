"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

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