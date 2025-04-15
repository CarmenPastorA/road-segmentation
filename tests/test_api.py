import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    with open("data/raw/valid/64_sat.jpg", "rb") as img:
        response = client.post("/predict", files={"file": img})
    assert response.status_code == 200
    assert "mask" in response.json()
