FROM ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc AS uv
FROM python:3.14.7-slim-bookworm@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-cache
COPY . .
RUN useradd --uid 10001 --user-group --create-home --home-dir /home/appuser \
      --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/raw /app/results /app/models /app/runs /app/outputs \
    && chown -R 10001:10001 /app/data /app/results /app/models /app/runs /app/outputs
ENV HOME=/home/appuser
USER 10001:10001
CMD ["python", "train_deep_pipeline.py", "--help"]
