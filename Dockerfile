FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY .streamlit ./.streamlit

EXPOSE 8000 8501
CMD ["uvicorn", "ecom.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
