FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 fastapi \
    && mkdir -p /data/umi-gpu \
    && chown -R fastapi:fastapi /app /data/umi-gpu
USER fastapi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
