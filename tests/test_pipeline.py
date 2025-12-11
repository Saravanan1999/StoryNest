from unittest.mock import MagicMock, patch

from storynest.pipeline import StoryNest, generate_bedtime_story, run_judge


@patch("storynest.pipeline.call_model")
def test_generate_bedtime_story_runs_full_pipeline(mock_call_model: MagicMock) -> None:
    """
    Ensure generate_bedtime_story wires together all agents correctly.

    We simulate:
    - SafetyGuard -> returns safe request
    - StoryPlanner -> returns plan
    - StoryGenerator -> returns initial story
    - Judge (1) -> low score to trigger refinement
    - Refinement -> returns improved story
    - Judge (2) -> high score to stop loop
    """
    safe_request = "Safe request for a bedtime story."
    story_plan = "TITLE: Safe Story"
    initial_story = "Initial story text."
    refined_story = "Refined story text."

    judge_low = "OVERALL_SCORE: 7.0\nFEEDBACK:\n- Too short"
    judge_high = "OVERALL_SCORE: 9.0\nFEEDBACK:\n- Great"

    mock_call_model.side_effect = [
        safe_request,   # SafetyGuard
        story_plan,     # StoryPlanner
        initial_story,  # StoryGenerator
        judge_low,      # Judge iteration 1
        refined_story,  # Refinement
        judge_high,     # Judge iteration 2
    ]

    result = generate_bedtime_story("raw user request", min_score=8.5, max_iterations=2)

    assert result == refined_story
    # We expect 6 calls as per the side_effect setup.
    assert mock_call_model.call_count == 6


@patch("storynest.pipeline.call_model")
def test_run_judge_parses_score_and_falls_back(mock_call_model: MagicMock) -> None:
    """
    Directly test run_judge's score parsing, including the fallback path.
    """
    # First call: well-formed judge output
    mock_call_model.side_effect = [
        "OVERALL_SCORE: 8.0\nFEEDBACK:\n- Great",
        "FEEDBACK ONLY WITHOUT SCORE",
    ]

    score, _ = run_judge("req", "plan", "story text")
    assert score == 8.0

    # Second call: missing OVERALL_SCORE should yield 0.0
    score2, _ = run_judge("req", "plan", "story text")
    assert score2 == 0.0


@patch("storynest.pipeline.run_judge")
@patch("storynest.pipeline.run_story_generator")
@patch("storynest.pipeline.run_story_planner")
@patch("storynest.pipeline.run_safety_guard")
def test_storynest_class_generate_method_uses_config(
    mock_safety: MagicMock,
    mock_planner: MagicMock,
    mock_generator: MagicMock,
    mock_judge: MagicMock,
) -> None:
    """
    Ensure StoryNest.generate_bedtime_story uses its config for loop control.
    """
    mock_safety.return_value = "safe"
    mock_planner.return_value = "plan"
    mock_generator.return_value = "story"
    # Always return high score to stop after first iteration
    mock_judge.return_value = (9.5, "OVERALL_SCORE: 9.5")

    pipeline = StoryNest()
    result = pipeline.generate_bedtime_story("raw")

    assert result == "story"
    mock_safety.assert_called_once()
    mock_planner.assert_called_once()
    mock_generator.assert_called_once()
    mock_judge.assert_called_once()



