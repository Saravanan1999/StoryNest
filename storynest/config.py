from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for a single agent call."""

    max_tokens: int
    temperature: float


@dataclass
class StoryNestConfig:
    """
    Configuration for the StoryNest pipeline.

    This makes it easy to tune thresholds, token budgets, and temperatures
    without changing core orchestration code.
    """

    min_score: float = 8.5
    max_iterations: int = 2

    # Use default_factory to avoid sharing mutable defaults between instances.
    safety_guard: AgentConfig = field(
        default_factory=lambda: AgentConfig(max_tokens=200, temperature=0.1)
    )
    story_planner: AgentConfig = field(
        default_factory=lambda: AgentConfig(max_tokens=400, temperature=0.5)
    )
    story_generator: AgentConfig = field(
        default_factory=lambda: AgentConfig(max_tokens=1200, temperature=0.7)
    )
    judge: AgentConfig = field(
        default_factory=lambda: AgentConfig(max_tokens=500, temperature=0.2)
    )
    refinement: AgentConfig = field(
        default_factory=lambda: AgentConfig(max_tokens=1200, temperature=0.6)
    )


__all__ = ["AgentConfig", "StoryNestConfig"]


