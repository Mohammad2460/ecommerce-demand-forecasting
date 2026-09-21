export PYTHONPATH := src

.PHONY: install data sample pipeline notebooks test lint api dashboard all

install:
	uv sync

data:
	mkdir -p data/raw/olist
	curl -L -o data/raw/olist.zip https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce
	unzip -o data/raw/olist.zip -d data/raw/olist && rm data/raw/olist.zip

sample:
	uv run python scripts/make_sample_data.py

pipeline:
	uv run python scripts/run_pipeline.py

notebooks:
	uv run python scripts/build_notebooks.py

all: install data pipeline notebooks

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

api:
	uv run uvicorn ecom.api.main:app --app-dir src --reload --port 8000

dashboard:
	uv run streamlit run src/ecom/dashboard/app.py
