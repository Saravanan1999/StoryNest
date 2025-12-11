from storynest.prompts import (
    build_judge_prompt,
    build_refinement_prompt,
    build_safety_guard_prompt,
    build_story_generator_prompt,
    build_story_planner_prompt,
)


def test_safety_guard_prompt_contains_original_request() -> None:
    req = "A spooky story"
    prompt = build_safety_guard_prompt(req)
    assert "Original request:" in prompt
    assert req in prompt


def test_story_planner_prompt_structure() -> None:
    safe_req = "A gentle bedtime story"
    prompt = build_story_planner_prompt(safe_req)
    for section in ["TITLE", "TARGET_AGE_RANGE", "MAIN_CHARACTERS", "STORY_ARC", "THEME_OR_MORAL"]:
        assert section in prompt
    assert safe_req in prompt


def test_story_generator_prompt_includes_plan() -> None:
    safe_req = "A forest adventure"
    plan = "TITLE: The Forest Adventure"
    prompt = build_story_generator_prompt(safe_req, plan)
    assert safe_req in prompt
    assert plan in prompt
    assert "Now write the full story." in prompt


def test_judge_prompt_rubric_present() -> None:
    safe_req = "A story about sharing"
    plan = "TITLE: Sharing is Caring"
    story = "Once upon a time..."
    prompt = build_judge_prompt(safe_req, plan, story)
    for word in ["OVERALL_SCORE", "FEEDBACK:", "STORY:", "SAFE_REQUEST:", "STORY_PLAN:"]:
        assert word in prompt


def test_refinement_prompt_mentions_feedback() -> None:
    safe_req = "A bedtime story"
    plan = "TITLE: Sleepy Time"
    story = "Once upon a time..."
    feedback = "OVERALL_SCORE: 7.5"
    prompt = build_refinement_prompt(safe_req, plan, story, feedback)
    assert "JUDGE_FEEDBACK" in prompt
    assert feedback in prompt



