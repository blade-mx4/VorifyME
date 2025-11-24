# Use a lightweight Python version
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# ------------------------------------------------
# INSTALL SYSTEM DEPENDENCIES (The Fix)
# We install ffmpeg (for pydub) and libsndfile (for librosa)
# ------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Start the server (This fixes the 'import string' error)
# ASSUMPTION: Your python file is named 'main.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
