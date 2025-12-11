"""
Prompt builders for the StoryNest agents.

Having these in a separate module makes it easier to:
- Reuse patterns
- Run prompt-focused experiments
- Keep the pipeline orchestration readable
"""


def build_safety_guard_prompt(user_request: str) -> str:
    return f"""
You are SafetyGuard, a safety and age-appropriateness filter for bedtime stories for children ages 5–10.

Given the user's original request, you MUST:
- Check for unsafe, violent, sexual, hateful, or otherwise inappropriate content.
- If the request is unsafe, gently rewrite it into a safe version that preserves the spirit of the request.
- If the request is already safe, keep it mostly the same but you may clarify it slightly.

Return ONLY the final, safe request text, with no explanations.

Original request:
{user_request}
"""


def build_story_planner_prompt(safe_request: str) -> str:
    return f"""
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


def build_story_generator_prompt(safe_request: str, story_plan: str) -> str:
    return f"""
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


def build_judge_prompt(safe_request: str, story_plan: str, story: str) -> str:
    return f"""
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


def build_refinement_prompt(
    safe_request: str,
    story_plan: str,
    story: str,
    judge_feedback: str,
) -> str:
    return f"""
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


__all__ = [
    "build_safety_guard_prompt",
    "build_story_planner_prompt",
    "build_story_generator_prompt",
    "build_judge_prompt",
    "build_refinement_prompt",
]


