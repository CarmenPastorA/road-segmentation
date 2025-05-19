import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch
from api.main import app

os.environ["SKIP_MODEL_LOADING"] = "1"

client = TestClient(app)
TEST_IMAGE_PATH = "tests/test.jpg"

@pytest.mark.asyncio
@patch("api.model_loader.load_model")
async def test_predict_image_valid(mock_load_model):
    mock_load_model.return_value = (None, None)
    dummy_mask = torch.zeros((1, 1, 256, 256))

    assert os.path.exists(TEST_IMAGE_PATH), f"Test image {TEST_IMAGE_PATH} not found"
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    with patch("api.main.model", lambda x: dummy_mask):
        with open(TEST_IMAGE_PATH, "rb") as img:
            response = client.post("/predict-image", files={"file": ("test.jpg", img, "image/jpeg")})

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.headers["content-type"] == "image/png"

    img = Image.open(io.BytesIO(response.content))
    assert img.format == "PNG"
    assert img.size == (256, 256), f"Expected mask size (256, 256), got {img.size}"

@pytest.mark.asyncio
@patch("api.model_loader.load_model")
async def test_predict_binary_valid(mock_load_model):
    mock_load_model.return_value = (None, None)
    dummy_mask = torch.zeros((1, 1, 256, 256))

    assert os.path.exists(TEST_IMAGE_PATH), f"Test image {TEST_IMAGE_PATH} not found"
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")

    with patch("api.main.model", lambda x: dummy_mask):
        with open(TEST_IMAGE_PATH, "rb") as img:
            response = client.post("/predict-binary", files={"file": ("test.jpg", img, "image/jpeg")})

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "mask" in response.json(), f"Response missing 'mask': {response.text}"
    mask = response.json()["mask"]
    assert isinstance(mask, list)
    assert len(mask) == 256
    assert all(len(row) == 256 for row in mask)

@pytest.mark.asyncio
async def test_predict_image_invalid():
    invalid_path = "tests/invalid.txt"
    with open(invalid_path, "w") as f:
        f.write("not an image")
    with open(invalid_path, "rb") as invalid:
        response = client.post("/predict-image", files={"file": ("invalid.txt", invalid, "text/plain")})
    assert response.status_code == 422
    os.remove(invalid_path)

@pytest.mark.asyncio
async def test_predict_binary_empty():
    empty_path = "tests/empty.jpg"
    with open(empty_path, "wb") as empty:
        pass
    with open(empty_path, "rb") as empty:
        response = client.post("/predict-binary", files={"file": ("empty.jpg", empty, "image/jpeg")})
    assert response.status_code == 422
    os.remove(empty_path)
