#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_inference_api.py: Tests para la API de segmentación de carreteras.

⚠️ Requiere que el servidor FastAPI esté corriendo en http://localhost:8000.
Por ejemplo, ejecutando: uvicorn app.main:app --reload
"""

import os
import time
import requests
from PIL import Image
import io
from pyproj import Transformer
from io import BytesIO

# Configuración de la URL base de la API
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

TEST_IMAGES_DIR = "../data/testing"
TEST_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "test_image.png")
INVALID_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "invalid_image.txt")



def check_server(api_url=API_URL, retries=3, delay=5, timeout=2):
    print(f"🔍 Verificando disponibilidad del servidor en {api_url}...")
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{api_url}/docs", timeout=timeout)
            if response.status_code == 200:
                print("✅ El servidor está disponible.")
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    raise RuntimeError(f"🚨 El servidor no respondió correctamente tras {retries} intentos.")


def test_upload_image_success():
    """Test para la subida de una imagen válida y segmentación exitosa."""
    with open(TEST_IMAGE_PATH, "rb") as img_file:
        files = {"file": ("test_image.png", img_file, "image/png")}
        response = requests.post(f"{API_URL}/predict-image", files=files)
        print("✅ Prueba de subida de imagen exitosa.")
    assert response.status_code == 200
    assert response.content, "La respuesta no contiene una imagen."
    try:
        Image.open(io.BytesIO(response.content))
    except Exception:
        assert False, "La respuesta no es una imagen válida."


def test_upload_invalid_image():
    """Test para la subida de un archivo no válido (no imagen)."""
    with open(INVALID_IMAGE_PATH, "rb") as invalid_file:
        files = {"file": ("invalid_image.txt", invalid_file, "text/plain")}
        response = requests.post(f"{API_URL}/predict-image", files=files)
        print("✅ Prueba de subida de imagen inválida.")
    
    assert response.status_code == 422

def download_wms_image(bbox_latlon):
    """Función para descargar imagen WMS y convertirla en PIL Image"""

    # Convert bbox from EPSG:4326 to EPSG:3857
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(bbox_latlon[0], bbox_latlon[1])
    x_max, y_max = transformer.transform(bbox_latlon[2], bbox_latlon[3])
    bbox_3857 = [x_min, y_min, x_max, y_max]

    wms_url = "https://www.ign.es/wms-inspire/pnoa-ma"
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "OI.OrthoimageCoverage",
        "bbox": ",".join(map(str, bbox_3857)),
        "width": 512,
        "height": 512,
        "crs": "EPSG:3857",
        "format": "image/jpeg"
    }

    response = requests.get(wms_url, params=params, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

def test_select_coordinates_success():
    """Test para la selección de coordenadas y descarga de imagen usando función local."""
    bbox_latlon = [-3.7, 40.0, -3.6, 40.1]

    # Usar función directamente para descargar la imagen
    img = download_wms_image(bbox_latlon)
    print("✅ Prueba de descarga de imagen WMS exitosa.")

    assert img is not None, "No se pudo descargar la imagen WMS"
    try:
        img.verify()  # Verifica que es una imagen válida
    except Exception:
        assert False, "La imagen descargada no es válida"


if __name__ == "__main__":
    print("🧪 Verificando servidor...")
    check_server()
    print("✅ Ejecutando tests...")
    test_upload_image_success()
    test_upload_invalid_image()
    test_select_coordinates_success()
    print("🎉 Todos los tests han pasado correctamente.")
