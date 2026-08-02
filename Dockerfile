# ─── MTBot Dockerfile ───────────────────────────────────────────────────────────
# Produces a lean, reproducible image of the trading bot.
# Build:  docker build -t mtbot .
# Run:    docker compose up -d   (see docker-compose.yml)

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Install Python dependencies first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source
COPY . .

# Create the logs directory so the log writer never fails
RUN mkdir -p logs

# Expose the Web UI port (must match WEB_PORT in .env)
EXPOSE 6594

# Default command: start the bot
CMD ["python3", "main.py"]
