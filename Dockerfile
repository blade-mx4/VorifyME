# Use a lightweight Python version
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (FFMPEG is required for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Start the server
# "app:app" means: look in 'app.py' for the 'app' object
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
