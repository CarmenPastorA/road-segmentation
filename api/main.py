import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

from api.model_loader import load_model
from api.utils import preprocess_image, postprocess_mask

app = FastAPI(title="Road Segmentation API")

# Load model unless testing
model = device = None
if not os.environ.get("SKIP_MODEL_LOADING"):
    model, device = load_model()

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid image file")

    input_tensor = preprocess_image(image, device)
    output = model(input_tensor)
    mask_np = postprocess_mask(output)

    _, buffer = cv2.imencode(".png", mask_np * 255)
    return Response(content=buffer.tobytes(), media_type="image/png")

@app.post("/predict-binary")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid image file")

    input_tensor = preprocess_image(image, device)
    mask = model(input_tensor)
    mask_np = postprocess_mask(mask)

    return {"mask": mask_np.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
