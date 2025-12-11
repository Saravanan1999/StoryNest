## StoryNest

Interactive story generator powered by OpenAI.

## Prerequisites

- **Python**: 3.9+ installed (`python3 --version`)
- **OpenAI API key**: An API key with access to chat models

## Setup with virtual environment (recommended)

From the project root (`/Users/stygianphantom/Documents/AI Agent Deployment Engineer Takehome`):

```bash
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure your OpenAI API key

### Option 1: Use a `.env` file (recommended)

1. In the project root (`/Users/stygianphantom/Documents/AI Agent Deployment Engineer Takehome`), create a file named `.env`:

   ```env
   OPENAI_API_KEY=your_real_api_key_here
   ```

2. With your virtual environment activated, make sure dependencies are installed (including `python-dotenv`):

   ```bash
   pip install -r requirements.txt
   ```

`main.py` will automatically load this `.env` file via `python-dotenv` and read `OPENAI_API_KEY`.

### Option 2: Set the environment variable in your shell

Set the `OPENAI_API_KEY` environment variable in your shell before running the app. For example on macOS with zsh:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can add that line to your `~/.zshrc` so it’s available in every new terminal session.

## Run the project

With the virtual environment activated and `OPENAI_API_KEY` set:

```bash
python main.py
```

The script will prompt:

```text
What kind of story do you want to hear?
```

Type a description (for example: `A cozy mystery in a small seaside town`) and press Enter to generate a story.

## StoryNest architecture (high level)

- **Package layout**
  - `storynest/pipeline.py`: multi-agent orchestration (`StoryNest` class, `generate_bedtime_story`).
  - `storynest/prompts.py`: prompt builder helpers for each agent.
  - `storynest/config.py`: `StoryNestConfig` and per-agent settings (tokens, temperatures, thresholds).
  - `storynest/llm_client.py`: OpenAI + dotenv wiring and error handling.
  - `storynest/eval.py`: offline evaluation helpers.
- **Agents**
  - SafetyGuard → Story Planner → Story Generator → Judge (+ Refinement loop).
  - Logging at `INFO` level traces each step and LLM call.

## Offline eval / experiments (optional)

You can run lightweight offline evaluations and experiments from a Python shell or notebook:

```python
from storynest import StoryNestConfig, run_offline_eval, write_eval_results_jsonl

prompts = [
    "A gentle bedtime story about a kid who is nervous about starting school.",
    "A magical adventure in a friendly forest.",
]

config = StoryNestConfig(min_score=8.0, max_iterations=1)
results = run_offline_eval(prompts, config=config)
write_eval_results_jsonl(results, "evals/storynest_results.jsonl")
```

This will generate stories, have the Judge agent score them, and write structured outputs for further analysis.

Or, to run the bundled eval suite in one go:

```bash
python run_eval.py
```

This runs a small set of example prompts and writes results to `evals/storynest_results.jsonl`.

## Testing & coverage

With your virtual environment activated:

```bash
pytest
```

To see file-by-file coverage for the `storynest` package and `main.py`:

```bash
pytest --cov=storynest --cov=main --cov-report=term-missing
```

At the time of writing, the core StoryNest modules are covered at roughly **98%+** line coverage (with `pipeline.py`, `config.py`, and `prompts.py` at 100%).

## Next steps (for the assignment)

Before submitting the assignment, fill in the comment block at the top of `main.py` with a brief description of what you would have built next if you had 2 more hours to work on this project.