#test/test_api.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["SKIP_MODEL_LOADING"] = "1"

import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
import numpy as np
from PIL import Image
import io

client = TestClient(app)
TEST_IMAGE_PATH = "tests/test_resized.jpg"

@pytest.mark.asyncio
@patch("api.utils.predict")  # Mock the prediction function
@patch("api.model_loader.load_model")  # Mock model loading
async def test_predict_image_valid(mock_load_model, mock_predict):
    # Mock model and device
    mock_load_model.return_value = (None, None)
    # Mock prediction to return a dummy 256x256 binary mask
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    mock_predict.return_value = dummy_mask

    # Verify test image
    assert os.path.exists(TEST_IMAGE_PATH), f"Test image {TEST_IMAGE_PATH} not found"
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    assert img.mode == "RGB", "Test image must be RGB"
    
    with open(TEST_IMAGE_PATH, "rb") as img:
        response = client.post("/predict-image", files={"image": ("test_resized.jpg", img, "image/jpeg")})
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.headers["content-type"] == "image/png"
    
    img = Image.open(io.BytesIO(response.content))
    assert img.format == "PNG"
    assert img.size == (256, 256), f"Expected mask size (256, 256), got {img.size}"

@pytest.mark.asyncio
@patch("api.utils.predict")
@patch("api.model_loader.load_model")
async def test_predict_binary_valid(mock_load_model, mock_predict):
    mock_load_model.return_value = (None, None)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    mock_predict.return_value = dummy_mask

    assert os.path.exists(TEST_IMAGE_PATH), f"Test image {TEST_IMAGE_PATH} not found"
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    assert img.mode == "RGB", "Test image must be RGB"
    
    with open(TEST_IMAGE_PATH, "rb") as img:
        response = client.post("/predict-binary", files={"image": ("test_resized.jpg", img, "image/jpeg")})
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "mask" in response.json(), f"Response missing 'mask': {response.text}"
    mask = response.json()["mask"]
    assert isinstance(mask, list)
    assert len(mask) == 256, f"Expected mask height 256, got {len(mask)}"
    assert all(len(row) == 256 for row in mask), "Expected mask width 256"

@pytest.mark.asyncio
async def test_predict_image_invalid():
    invalid_path = "tests/invalid.txt"
    with open(invalid_path, "w") as f:
        f.write("This is not an image")
    with open(invalid_path, "rb") as invalid:
        response = client.post("/predict-image", files={"image": ("invalid.txt", invalid, "text/plain")})
    
    assert response.status_code in [400, 422], f"Expected 400 or 422, got {response.status_code}"
    os.remove(invalid_path)

@pytest.mark.asyncio
async def test_predict_binary_empty():
    empty_path = "tests/empty.jpg"
    with open(empty_path, "wb") as empty:
        pass
    with open(empty_path, "rb") as empty:
        response = client.post("/predict-binary", files={"image": ("empty.jpg", empty, "image/jpeg")})
    
    assert response.status_code == 422
    os.remove(empty_path)