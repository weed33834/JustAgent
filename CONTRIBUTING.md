# Contributing

Thanks for considering a contribution. Here is how things work.

## Before you start

- Search existing issues and pull requests to avoid duplicates.
- For anything larger than a typo fix, open an issue first to discuss the approach.

## Setup

```bash
git clone https://gitcode.com/badhope/justagent.git
cd justagent
uv sync --extra dev
```

This installs JustAgent in editable mode with all dev dependencies — Ruff, mypy, Pyright, pytest, and tooling.

## Making changes

1. Create a branch from `main`.
2. Make your changes.
3. Add or update tests in `tests/` if applicable.
4. Run the full check suite:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pytest tests/ -v
```

All checks must pass before a PR is reviewed.

## Code style

- Ruff handles formatting and linting — no manual style decisions needed.
- Type hints are required. MyPy and Pyright both run in strict mode.
- Docstrings follow Google style.
- Avoid adding new dependencies unless there is a strong reason. Discuss in the issue first.

## Commit messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

## Testing

- Unit tests go in `tests/unit/`.
- Integration tests go in `tests/integration/` and are marked with `@pytest.mark.integration`.
- Coverage must stay at or above 85%. Check with `uv run pytest --cov=justagent`.

## License

By contributing, you agree that your work will be licensed under the MIT License.
