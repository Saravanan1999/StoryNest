import logging
import re
from typing import Tuple

from .llm_client import call_model


logger = logging.getLogger(__name__)


class StoryNest:
    """
    Class-based pipeline for the StoryNest bedtime story generator.

    This groups the individual agent steps into a single, reusable object.
    """

    def __init__(self, min_score: float = 8.5, max_iterations: int = 2) -> None:
        self.min_score = min_score
        self.max_iterations = max_iterations

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

        for iteration in range(self.max_iterations):
            logger.info("Judge iteration %d started", iteration + 1)
            overall_score, feedback = self.run_judge(safe_request, story_plan, story)
            logger.info("Judge score after iteration %d: %.2f", iteration + 1, overall_score)

            if overall_score >= self.min_score:
                logger.info(
                    "Score threshold reached (>= %.2f); stopping refinement loop", self.min_score
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
    prompt = f"""
You are SafetyGuard, a safety and age-appropriateness filter for bedtime stories for children ages 5–10.

Given the user's original request, you MUST:
- Check for unsafe, violent, sexual, hateful, or otherwise inappropriate content.
- If the request is unsafe, gently rewrite it into a safe version that preserves the spirit of the request.
- If the request is already safe, keep it mostly the same but you may clarify it slightly.

Return ONLY the final, safe request text, with no explanations.

Original request:
{user_request}
"""
    logger.info("Calling SafetyGuard agent")
    result = call_model(prompt, max_tokens=200, temperature=0.1).strip()
    logger.info("SafetyGuard agent returned text of length %d", len(result))
    return result


def run_story_planner(safe_request: str) -> str:
    """
    Create a simple but explicit story plan for a bedtime story.

    The plan is returned as a structured text outline that the generator will follow.
    """
    prompt = f"""
You are a story planning assistant for children's bedtime stories (ages 5–10).

Based on the safe story request below, create a clear outline with these sections:
- TITLE
- TARGET_AGE_RANGE (must be 5–10)
- MAIN_CHARACTERS (bullet list with 1–3 sentence descriptions)
- SETTING
- STORY_ARC (Beginning, Middle, Climax, Resolution)
- THEME_OR_MORAL

Write in simple English that another model can easily follow.

SAFE_REQUEST:
{safe_request}
"""
    logger.info("Calling Story Planner agent")
    result = call_model(prompt, max_tokens=400, temperature=0.5).strip()
    logger.info("Story Planner agent returned text of length %d", len(result))
    return result


def run_story_generator(safe_request: str, story_plan: str) -> str:
    """
    Generate a complete story from the structured plan.
    """
    prompt = f"""
You are a warm, imaginative children's storyteller.
Your job is to write a single, self-contained bedtime story appropriate for children ages 5–10.

Follow the STORY_PLAN closely, including the characters, setting, and story arc.
Use simple, friendly language. Avoid anything scary, violent, or inappropriate.
Target length: roughly 600–1000 words.

SAFE_REQUEST:
{safe_request}

STORY_PLAN:
{story_plan}

Now write the full story.
"""
    logger.info("Calling Story Generator agent")
    result = call_model(prompt, max_tokens=1200, temperature=0.7).strip()
    logger.info("Story Generator agent returned text of length %d", len(result))
    return result


def run_judge(safe_request: str, story_plan: str, story: str) -> Tuple[float, str]:
    """
    Judge the quality and safety of the story.

    Returns:
        overall_score: float between 0 and 10
        feedback: textual critique / suggestions
    """
    prompt = f"""
You are a strict but fair judge of children's bedtime stories for ages 5–10.

Evaluate the STORY below using this rubric:
- Safety & age-appropriateness
- Coherence and clear beginning/middle/end
- Engagement and creativity
- Alignment with the SAFE_REQUEST and STORY_PLAN

Instructions:
- Give an OVERALL_SCORE between 0 and 10 (you may use decimals).
- Then provide 3–6 bullet points of FEEDBACK with concrete suggestions for improvement.

Return your answer in this exact format:
OVERALL_SCORE: <number>
FEEDBACK:
- ...
- ...

SAFE_REQUEST:
{safe_request}

STORY_PLAN:
{story_plan}

STORY:
{story}
"""
    logger.info("Calling Judge agent")
    judge_output = call_model(prompt, max_tokens=500, temperature=0.2).strip()
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
    prompt = f"""
You are a story editor improving bedtime stories for children ages 5–10.

You will receive:
- SAFE_REQUEST: what the child (or parent) asked for
- STORY_PLAN: the planned structure
- CURRENT_STORY: the story to improve
- JUDGE_FEEDBACK: a review with an OVERALL_SCORE and bullet points

Your job:
- Fix the issues mentioned in JUDGE_FEEDBACK.
- Keep the story safe, warm, and age-appropriate.
- Maintain the original intent of the SAFE_REQUEST and the structure of the STORY_PLAN.
- Only change what is necessary to address the feedback.

Return ONLY the revised story, with no commentary.

SAFE_REQUEST:
{safe_request}

STORY_PLAN:
{story_plan}

CURRENT_STORY:
{story}

JUDGE_FEEDBACK:
{judge_feedback}
"""
    logger.info("Calling Refinement agent")
    result = call_model(prompt, max_tokens=1200, temperature=0.6).strip()
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
    pipeline = StoryNest(min_score=min_score, max_iterations=max_iterations)
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


