# Contributing

Thanks for your interest in improving OmniGenesis.

## Before You Start

- Search existing issues before opening a new one.
- For new features, open an issue first to discuss scope and approach.
- Keep pull requests focused and small when possible.

## Local Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest ruff pre-commit
pre-commit install
```

## Development Workflow

1. Create a feature branch from `main`.
2. Make focused changes with clear commit messages.
3. Run checks locally:
   - `ruff check .`
   - `pytest`
4. Update docs/config examples if behavior changes.
5. Open a pull request with a clear summary and testing notes.

## Coding Guidelines

- Keep modules cohesive and responsibilities clear.
- Preserve thread safety around shared model operations.
- Keep config changes in `omnigenesis.yaml` rather than hardcoding values.
- Avoid introducing hidden side effects during module import.

## Pull Request Checklist

- Tests added/updated for new behavior.
- Lint/tests pass locally.
- README/config/docs updated where needed.
- No unrelated refactors bundled with the change.
