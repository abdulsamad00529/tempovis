.PHONY: install run test lint format type-check docker-up docker-down docker-logs clean

# ── Variables ──────────────────────────────────────────────────────────────────
PYTHON      := python
PIP         := pip
APP_MODULE  := app.main:app
PORT        ?= 8000

# ── Local development ──────────────────────────────────────────────────────────

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m uvicorn $(APP_MODULE) \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level info

# ── Quality ────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

lint:
	$(PYTHON) -m ruff check app/ tests/

format:
	$(PYTHON) -m ruff format app/ tests/

type-check:
	$(PYTHON) -m mypy app/

check: lint type-check test

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-restart:
	docker compose restart api

# ── Data ───────────────────────────────────────────────────────────────────────

generate-data:
	$(PYTHON) data/generate_samples.py

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
