# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including cmake and ninja-build
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install the required packages one by one to better handle dependencies
RUN pip install --no-cache-dir aiohttp
RUN pip install --no-cache-dir openai
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir newrelic
RUN pip install --no-cache-dir google-generativeai
RUN pip install --no-cache-dir google.ai.generativelanguage

# Copy the Python script into the container
COPY smtp_server.py .

# Make port 2525 available to the world outside this container
EXPOSE 2525

# Run smtp_server.py when the container launches
CMD ["newrelic-admin", "run-program", "python", "smtp_server.py"]
