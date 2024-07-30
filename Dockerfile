# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install the required packages
RUN pip install --no-cache-dir aiohttp openai

# Copy the Python script into the container
COPY smtp_server.py .

# Make port 2525 available to the world outside this container
EXPOSE 2525

# Run smtp_server.py when the container launches
CMD ["python", "smtp_server.py"]
