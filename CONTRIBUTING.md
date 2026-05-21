# Contributing to TempoVis

Thank you for your interest in contributing! This document covers how to set up
the dev environment, run tests, and submit changes.

---

## Dev Environment Setup

**Requirements:** Python 3.11+, Docker (for Postgres + Redis), Git.

```bash
# 1. Clone and enter the repo
git clone https://github.com/abdulsamad00529/tempovis
cd tempovis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies (including dev extras)
pip install -e ".[dev]"
# or if using requirements files:
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx ruff

# 4. Copy and configure environment
cp .env.example .env
# Set DRY_RUN=true in .env to skip real OpenAI calls during dev

# 5. Start backing services (Postgres + Redis only)
docker-compose up postgres redis -d
```

---

## Running Tests

```bash
# All tests (uses DRY_RUN mock — no real API calls needed)
pytest

# With coverage
pytest --cov=app --cov-report=html

# Single file
pytest tests/test_api.py -v

# Skip the HuggingFace data-loader tests (requires dataset download)
pytest --ignore=tests/test_data_loader.py
```

Tests use `httpx.AsyncClient` with `ASGITransport` — no live server needed.
Postgres and Redis dependencies are mocked via `pytest` fixtures.

---

## Code Style

We use **ruff** for linting and formatting.

```bash
# Check
ruff check .

# Auto-fix
ruff check . --fix

# Format
ruff format .
```

Configuration lives in `pyproject.toml`. CI will fail on any lint errors.

---

## Pull Request Guidelines

1. **Open an issue first** for anything beyond a trivial bug fix. Describe the
   problem and your proposed solution before writing code.

2. **One feature or fix per PR.** Split unrelated changes into separate PRs.

3. **Write tests** for new behaviour. The test suite must stay green.

4. **Keep commits clean.** Squash fixup commits before requesting review.

5. **Update docs** if your change affects the API, environment variables, or
   the architecture described in README.md.

6. **No secrets.** Never commit `.env`, API keys, passwords, or credentials.
   Use `.env.example` for documentation.

---

## Project Layout Quick Reference

| Path | Purpose |
|------|---------|
| `app/services/reasoner.py` | VLM call logic + DRY_RUN mock |
| `app/services/renderer.py` | Domain-aware plot rendering |
| `app/services/agent.py` | LangGraph agentic loop |
| `app/api/routes.py` | FastAPI route handlers |
| `app/core/config.py` | All settings (via pydantic-settings) |
| `dashboard/app.py` | Streamlit UI |
| `tests/` | pytest async integration tests |

---

## Reporting Bugs

Open a GitHub Issue with:
- What you expected to happen
- What actually happened (paste logs, not screenshots)
- Your OS, Python version, and relevant `.env` settings (redact keys)
