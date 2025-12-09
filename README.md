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

## Next steps (for the assignment)

Before submitting the assignment, fill in the comment block at the top of `main.py` with a brief description of what you would have built next if you had 2 more hours to work on this project.