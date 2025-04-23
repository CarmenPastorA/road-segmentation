# app/api_client.py

"""
api_client.py

This module provides a utility function to interact with the FastAPI backend
for road segmentation. It sends a satellite image to the `/predict-image` 
endpoint and returns the predicted segmentation mask as a PIL image.

Used by the Streamlit frontend (app/ui.py) to offload prediction to the API.

- send_image_for_prediction(image: PIL.Image, api_url: str):
    Sends an image via POST request to the FastAPI model server and 
    receives a predicted binary mask image in response.

"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from PIL import Image
import io

def send_image_for_prediction(image: Image.Image, api_url: str = "http://localhost:8000/predict-image"):
    """
    Sends a PIL image to the FastAPI server and returns a PIL image mask.
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        files = {"file": ("image.jpg", buffered.getvalue(), "image/jpeg")}

        response = requests.post(api_url, files=files, timeout=20)

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to contact prediction API: {e}")
