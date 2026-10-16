# AI/ML Course

A complete, self-paced AI/ML engineering course (~36 weeks at 10–15 hrs/week), generated once by
Claude and then studied, with Claude reviewing the assignments.

## Status

Set-up only. No course content has been generated yet.

## How to build the course

1. Install [uv](https://docs.astral.sh/uv/) (the Python package manager the labs use).
2. Open this folder in Claude Code and run the prompts in [`course/PROMPTS.md`](course/PROMPTS.md)
   in order: Step 1A (outline) → 1B (one per part) → 2 (style guide, glossary, manifest,
   dependencies) → 3 (one week at a time, repeated) → 4 (QA per part) → 5 (revision material).
3. `course/manifest.json` tracks what has been generated, so any session can pick up where the
   last one stopped.

## How to study

- Read the lesson in `lessons/`, run its lab in `labs/`, do the exercises in `my-work/`.
- Ask Claude: `Review my-work/<file>` and read the result in `reviews/`.
- End each week with its quiz in `quizzes/` and its cheat sheet in `cheatsheets/`.
- Don't open `answers/`. It holds the solutions Claude grades against.
