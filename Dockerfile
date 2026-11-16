FROM python:3.12-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY incidentforge ./incidentforge
COPY configs ./configs

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /install /usr/local
COPY configs ./configs
COPY schema ./schema

USER appuser

ENTRYPOINT ["incidentforge"]
CMD ["--help"]
