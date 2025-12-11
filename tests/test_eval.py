import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storynest.config import StoryNestConfig
from storynest.eval import (
    EvalResult,
    StoryEvalMetrics,
    run_offline_eval,
    score_story_with_metrics,
    write_eval_results_jsonl,
)


@patch("storynest.eval.call_model")
def test_score_story_with_metrics_parses_json(mock_call_model: MagicMock) -> None:
    payload = {
        "helpfulness": 4,
        "safety": 5,
        "coherence": 4,
        "verbosity": 3,
        "relevance": 5,
        "issues_reason": "",
        "goal_completion_score": 1.0,
        "goal_completion_reason": "Goal achieved",
        "conversation_score": 0.9,
        "final_score": 0.925,
    }
    mock_call_model.return_value = json.dumps(payload)

    metrics = score_story_with_metrics("req", "story")

    assert isinstance(metrics, StoryEvalMetrics)
    assert metrics.helpfulness == 4
    assert metrics.safety == 5
    assert metrics.goal_completion_score == 1.0


@patch("storynest.eval.call_model")
def test_score_story_with_metrics_bad_json_raises(mock_call_model: MagicMock) -> None:
    # Return a non-JSON string to trigger the JSONDecodeError path.
    mock_call_model.return_value = "NOT JSON"

    with pytest.raises(json.JSONDecodeError):
        score_story_with_metrics("req", "story")


@patch("storynest.eval.run_judge")
@patch("storynest.eval.run_story_generator")
@patch("storynest.eval.run_story_planner")
@patch("storynest.eval.run_safety_guard")
@patch("storynest.eval.score_story_with_metrics")
def test_run_offline_eval_uses_pipeline_and_metrics(
    mock_metrics: MagicMock,
    mock_safety: MagicMock,
    mock_planner: MagicMock,
    mock_generator: MagicMock,
    mock_judge: MagicMock,
) -> None:
    prompts = ["p1", "p2"]

    mock_safety.side_effect = ["safe1", "safe2"]
    mock_planner.side_effect = ["plan1", "plan2"]
    mock_generator.side_effect = ["story1", "story2"]
    mock_judge.side_effect = [(8.0, "JUDGE1"), (9.0, "JUDGE2")]

    metrics = StoryEvalMetrics(
        helpfulness=4,
        safety=5,
        coherence=4,
        verbosity=3,
        relevance=5,
        issues_reason="",
        goal_completion_score=1.0,
        goal_completion_reason="ok",
        conversation_score=0.9,
        final_score=0.925,
    )
    mock_metrics.side_effect = [metrics, metrics]

    cfg = StoryNestConfig(min_score=8.0, max_iterations=1)
    results = run_offline_eval(prompts, config=cfg)

    assert len(results) == 2
    for res in results:
        assert isinstance(res, EvalResult)
        assert res.metrics is not None


def test_write_eval_results_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    results = [
        EvalResult(
            prompt="p",
            safe_request="s",
            story_plan="plan",
            story="story",
            judge_score=8.0,
            judge_feedback="good",
            metrics=None,
        )
    ]

    write_eval_results_jsonl(results, path)
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["prompt"] == "p"



