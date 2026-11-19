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

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps .

EXPOSE 8000 8501 5000
