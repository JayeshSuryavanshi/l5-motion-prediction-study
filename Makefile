L5KIT_VERSION := 1.5.0

.PHONY: setup test lint format

setup:
	uv sync
	uv pip install --no-deps l5kit==$(L5KIT_VERSION)
	uv run python scripts/patch_l5kit.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts
