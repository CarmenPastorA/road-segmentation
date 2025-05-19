# Use a slim version of Python 3.10 as base image
FROM python:3.10-slim

# Install necessary system packages (compiler and image processing libs)
RUN apt-get update && apt-get install -y gcc libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the full project into the image
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose both FastAPI (8000) and Streamlit (7860) ports
EXPOSE 8000 7860

# Copy the startup script that will launch both services
COPY deployment/startup.sh /startup.sh

# Make the script executable
RUN chmod +x /startup.sh

# Set the startup command
CMD ["/startup.sh"]
