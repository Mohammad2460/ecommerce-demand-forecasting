export PYTHONPATH := src

.PHONY: install sample pipeline test lint api dashboard

install:
	uv sync

sample:
	uv run python scripts/make_sample_data.py

pipeline:
	uv run python scripts/run_pipeline.py

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

api:
	uv run uvicorn ecom.api.main:app --reload --port 8000

dashboard:
	uv run streamlit run src/ecom/dashboard/app.py
