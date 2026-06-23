# Use official lightweight Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies if any are needed (e.g., git, gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Install the project in editable mode so setup.py routes imports correctly
RUN pip install -e .

# Expose default Flask port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]
