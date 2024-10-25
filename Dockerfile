FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.4.26/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.cargo/bin/:$PATH"

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --frozen

# Copy the Python script into the container
COPY smtp_server.py .

# Make port 2525 available to the world outside this container
EXPOSE 2525

# Run smtp_server.py when the container launches
CMD ["newrelic-admin", "run-program", "python", "smtp_server.py"]


