# Stage 1: Build the environment with all dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies if any C-extensions need compiling
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install packages
# This is done first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---
# Stage 2: Create the final, lean production image
FROM python:3.11-slim

# --- THE FIX IS HERE: Install the missing system library for LightGBM ---
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed packages AND executables from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy our application code and model artifacts
COPY src/ src/
COPY models/ models/

# Expose the port the API will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]