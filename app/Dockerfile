# Use minimal Python base image
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y gcc libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install only required packages for the UI
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app code (ui.py + api_client.py)
COPY app/ ./app

# Expose the default Streamlit port
EXPOSE 7860

# Start the Streamlit app
CMD ["streamlit", "run", "app/ui.py", "--server.port=7860", "--server.address=0.0.0.0"]
