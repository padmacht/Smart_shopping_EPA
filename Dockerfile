FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src:/app/app \
    SHOPSMART_ROOT=/app

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY app ./app
COPY api ./api
COPY configs ./configs
# Data, trained models and evaluation reports are baked into the image so it runs
# on its own in the cloud (Render). docker compose still mounts the local folders over them.
COPY data ./data
COPY models ./models
COPY reports ./reports

# requirements.txt ends with "-e ." which installs the shopsmart package from src/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501 5000

# Default: start the API. Render sets $PORT; docker compose overrides this per service.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
