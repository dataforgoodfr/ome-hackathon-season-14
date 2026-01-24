# Install uv
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

# Copy project files first
COPY pyproject.toml uv.lock ./
COPY sentiment ./key-words

# Install dependencies and project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM python:3.12-slim

WORKDIR /app

# Copy the environment and source code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/key-words /app/key-words

# Expose port
EXPOSE 8003

# Run the service
CMD ["/app/.venv/bin/python", "key-words/api.py"]
