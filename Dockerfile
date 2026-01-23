ARG PYTHON_VERSION

# --------------------------------------------------
#       Base image
# --------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ARG APP_NAME
ARG APP_VERSION
ARG COMMIT_HASH

ARG USERNAME=${APP_NAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Required by GHCR
LABEL maintainer="Alexandre Brasseur"
LABEL org.opencontainers.image.source="https://github.com/abrasseu/ome-hackathon-season-14"
LABEL org.opencontainers.image.version=${APP_VERSION}
LABEL org.opencontainers.image.revision=${COMMIT_HASH}

# Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 0

# Create the user
RUN mkdir -p "/app" \
    && groupadd -g "${USER_GID}" "${USERNAME}" \
    && useradd -l -r -d "/app" -u "${USER_UID}" -g "${USERNAME}" "${USERNAME}" \
    && chown -R "${USERNAME}:${USERNAME}" "/app" \
    && rm -rf "/var/log" "/var/cache"

WORKDIR /app

# --------------------------------------------------
#       Builder image
# --------------------------------------------------

FROM base AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /build

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=true
ENV VIRTUAL_ENV=/build/.venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Install requirements first (less likely to change)
COPY --link ./pyproject.toml ./uv.lock ./
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    uv lock --check \
    && uv sync --frozen --no-default-groups --no-editable --no-install-project

# Finally install application
COPY --link ./README.md ./
COPY --link ./src ./src
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable \
    && uv pip check

# --------------------------------------------------
#       Production image
# --------------------------------------------------

FROM base AS app
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME
ARG USER_UID
ARG USER_GID

WORKDIR /app
USER ${USERNAME}

ENV PATH="/app/.venv/bin:$PATH"

COPY --chown=${USER_UID}:${USER_GID} --from=builder /build/.venv /app/.venv

RUN python -m reporto
