#!/bin/bash

# Start the FastAPI server in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds to ensure API is ready
sleep 5

# Start the Streamlit frontend in the foreground, pointing to the local API
API_URL=http://localhost:8000 streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0
