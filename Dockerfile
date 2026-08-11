# JustAgent Web — production container image.
# Build:  docker build -t justagent-web .
# Run:    docker run -p 8000:8000 \
#           -e JUSTAGENT_WEB_TOKEN=<token> \
#           -e OPENAI_API_KEY=<key> \
#           justagent-web

FROM python:3.12-slim

# uv for fast, reproducible dependency installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install the project (workspace incl. justagent-sdk) without the dev extras.
COPY pyproject.toml uv.lock README.md ./
COPY justagent-sdk ./justagent-sdk
RUN uv sync --no-dev --no-install-project --frozen

# Copy source
COPY src ./src
RUN uv sync --no-dev --frozen

# Runtime user
RUN useradd -m justagent
USER justagent

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000

# Start the JustAgent Web console.
CMD ["sh", "-c", "uv run python -m justagent web --host $HOST --port $PORT"]
