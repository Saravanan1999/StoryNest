import logging
import re
from typing import Tuple

from .config import StoryNestConfig
from .llm_client import call_model
from .prompts import (
    build_judge_prompt,
    build_refinement_prompt,
    build_safety_guard_prompt,
    build_story_generator_prompt,
    build_story_planner_prompt,
)


logger = logging.getLogger(__name__)


class StoryNest:
    """
    Class-based pipeline for the StoryNest bedtime story generator.

    This groups the individual agent steps into a single, reusable object.
    """

    def __init__(self, config: StoryNestConfig | None = None) -> None:
        """
        Create a StoryNest pipeline.

        A custom StoryNestConfig can be supplied for experiments; by default
        we use sensible values tuned for this assignment.
        """
        self.config = config or StoryNestConfig()

    # Thin wrappers for each agent stage so they can be overridden or extended later.
    def run_safety_guard(self, user_request: str) -> str:
        return run_safety_guard(user_request)

    def run_story_planner(self, safe_request: str) -> str:
        return run_story_planner(safe_request)

    def run_story_generator(self, safe_request: str, story_plan: str) -> str:
        return run_story_generator(safe_request, story_plan)

    def run_judge(self, safe_request: str, story_plan: str, story: str) -> Tuple[float, str]:
        return run_judge(safe_request, story_plan, story)

    def run_refinement(
        self,
        safe_request: str,
        story_plan: str,
        story: str,
        judge_feedback: str,
    ) -> str:
        return run_refinement(safe_request, story_plan, story, judge_feedback)

    def generate_bedtime_story(self, user_request: str) -> str:
        """
        Execute the full multi-agent pipeline to produce a final story.
        """
        logger.info("StoryNest pipeline started")
        logger.info("User request received")

        safe_request = self.run_safety_guard(user_request)
        logger.info("SafetyGuard agent completed")

        story_plan = self.run_story_planner(safe_request)
        logger.info("Story Planner agent completed")

        story = self.run_story_generator(safe_request, story_plan)
        logger.info("Story Generator agent completed")

        for iteration in range(self.config.max_iterations):
            logger.info("Judge iteration %d started", iteration + 1)
            overall_score, feedback = self.run_judge(safe_request, story_plan, story)
            logger.info("Judge score after iteration %d: %.2f", iteration + 1, overall_score)

            if overall_score >= self.config.min_score:
                logger.info(
                    "Score threshold reached (>= %.2f); stopping refinement loop", self.config.min_score
                )
                break

            logger.info("Refinement iteration %d started", iteration + 1)
            story = self.run_refinement(safe_request, story_plan, story, feedback)
            logger.info("Refinement iteration %d completed", iteration + 1)

        logger.info("StoryNest pipeline finished")
        return story


def run_safety_guard(user_request: str) -> str:
    """
    Ensure the request is safe and age-appropriate for kids 5–10.
    Returns a possibly rewritten, safe version of the request.
    """
    prompt = build_safety_guard_prompt(user_request)
    logger.info("Calling SafetyGuard agent")
    cfg = StoryNestConfig().safety_guard
    result = call_model(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature).strip()
    logger.info("SafetyGuard agent returned text of length %d", len(result))
    return result


def run_story_planner(safe_request: str) -> str:
    """
    Create a simple but explicit story plan for a bedtime story.

    The plan is returned as a structured text outline that the generator will follow.
    """
    prompt = build_story_planner_prompt(safe_request)
    logger.info("Calling Story Planner agent")
    cfg = StoryNestConfig().story_planner
    result = call_model(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature).strip()
    logger.info("Story Planner agent returned text of length %d", len(result))
    return result


def run_story_generator(safe_request: str, story_plan: str) -> str:
    """
    Generate a complete story from the structured plan.
    """
    prompt = build_story_generator_prompt(safe_request, story_plan)
    logger.info("Calling Story Generator agent")
    cfg = StoryNestConfig().story_generator
    result = call_model(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature).strip()
    logger.info("Story Generator agent returned text of length %d", len(result))
    return result


def run_judge(safe_request: str, story_plan: str, story: str) -> Tuple[float, str]:
    """
    Judge the quality and safety of the story.

    Returns:
        overall_score: float between 0 and 10
        feedback: textual critique / suggestions
    """
    prompt = build_judge_prompt(safe_request, story_plan, story)
    logger.info("Calling Judge agent")
    cfg = StoryNestConfig().judge
    judge_output = call_model(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature).strip()
    logger.info("Judge agent returned text of length %d", len(judge_output))

    # Parse overall score from the judge's output.
    score_match = re.search(r"OVERALL_SCORE:\s*([0-9]+(?:\.[0-9]+)?)", judge_output)
    if score_match:
        score = float(score_match.group(1))
    else:
        # Fallback if parsing fails; be conservative.
        logger.info("Could not parse judge score, defaulting to 0.0")
        score = 0.0

    logger.info("Parsed judge score: %.2f", score)
    return score, judge_output


def run_refinement(
    safe_request: str,
    story_plan: str,
    story: str,
    judge_feedback: str,
) -> str:
    """
    Refine the story based on the judge's feedback while preserving safety and intent.
    """
    prompt = build_refinement_prompt(safe_request, story_plan, story, judge_feedback)
    logger.info("Calling Refinement agent")
    cfg = StoryNestConfig().refinement
    result = call_model(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature).strip()
    logger.info("Refinement agent returned text of length %d", len(result))
    return result


def generate_bedtime_story(user_request: str, min_score: float = 8.5, max_iterations: int = 2) -> str:
    """
    Functional convenience wrapper around the class-based StoryNest pipeline.

    1. SafetyGuard Agent: sanitize / adjust the request.
    2. Story Planner Agent: outline the story.
    3. Story Generator Agent: draft the story.
    4. Judge Agent: evaluate quality and safety.
    5. Refinement Agent: improve the story based on judge feedback (loop).
    """
    cfg = StoryNestConfig(min_score=min_score, max_iterations=max_iterations)
    pipeline = StoryNest(config=cfg)
    return pipeline.generate_bedtime_story(user_request)


__all__ = [
    "StoryNest",
    "generate_bedtime_story",
    "run_safety_guard",
    "run_story_planner",
    "run_story_generator",
    "run_judge",
    "run_refinement",
]


