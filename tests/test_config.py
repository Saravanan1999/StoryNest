import pytest

from storynest.config import AgentConfig, StoryNestConfig


def test_storynest_config_defaults() -> None:
    cfg = StoryNestConfig()

    assert cfg.min_score == 8.5
    assert cfg.max_iterations == 2

    assert cfg.safety_guard.max_tokens == 200
    assert cfg.safety_guard.temperature == 0.1

    assert cfg.story_planner.max_tokens == 400
    assert cfg.story_generator.max_tokens == 1200
    assert cfg.judge.max_tokens == 500
    assert cfg.refinement.max_tokens == 1200


def test_storynest_config_mutation_does_not_affect_others() -> None:
    cfg1 = StoryNestConfig()
    cfg2 = StoryNestConfig()

    # mutate cfg1's safety_guard and ensure cfg2 is unchanged
    cfg1.safety_guard.max_tokens = 999

    assert cfg1.safety_guard.max_tokens == 999
    assert cfg2.safety_guard.max_tokens == 200


def test_agent_config_repr() -> None:
    cfg = AgentConfig(max_tokens=100, temperature=0.5)
    # basic sanity check that dataclass repr includes field names
    s = repr(cfg)
    assert "max_tokens" in s and "temperature" in s



