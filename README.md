# Road Segmentation from Satellite Images 🚧🛰️

This project is an end-to-end proof of concept for segmenting roads from satellite images using the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/data).

## 🔍 Objective

- Build a baseline segmentation model
- Create an API for serving predictions
- Build a simple UI for uploading images
- Track experiments with Weights & Biases
- Package everything in Docker for easy deployment
- Deploy in Azure

## 🚀 Structure

The project is organized as follows:

```
road-segmentation/
├── api/                        # API logic (FastAPI/Flask)
│   ├── main.py                 # Entrypoint for serving the model
│   ├── model_loader.py         # Load model & weights
│   └── utils.py                # Pre/post-processing helpers
│
├── app/                        # UI for uploading images and showing results
│   └── ui.py                   # Gradio or Streamlit app
│   └── api_client.py           # UI - API communication 
│
├── config/                     # Configuration files
│   └── config.yaml             # W&B, training and API configs
│
├── data/
│   ├── raw/                    # Original DeepGlobe dataset files
│   ├── processed/              # Cleaned or preprocessed data
│   ├── sample_train/           # Small subset of images and masks used for baseline
│   ├── dataset.py              # PyTorch Dataset class for image-mask pairs
│   ├── transforms.py           # Albumentations transforms for training/validation
│   ├── example_dataloader.py   # Standalone script to preview dataloader behavior
│  
├── docker/
│   └── Dockerfile              # Containerization instructions
│
├── models/
│   ├── mini_unet.py            # Mini U-Net baseline model
│   └── best_mini_unet.pth      # Trained weights (not pushed to repo)
│
├── notebooks/
│
├── scripts/
│   ├── create_subset.py       # Extracts a smaller training set (e.g. 300 samples)
│   ├── train.py               # Trains the segmentation model and logs metrics
│   ├── evaluate.py            # Loads model and evaluates on validation/test set
│   ├── predict.py             # Inference script for a single image using saved model
│
├── shared/
│
├── tests/
│   └── test_api.py             # API tests
│
├── wandb/                      # Experiment logs (auto-generated, .gitignored)
│
├── .dockerignore
├── .gitignore
├── README.md
├── requirements.txt
```

## 🧠 Model Architectures & Training

### Baseline: Mini U-Net
The baseline architecture is a custom **Mini U-Net** designed for efficient semantic segmentation. The network follows a classic encoder–decoder structure with skip connections and is trained on 300 samples:

- Three encoding blocks (`DoubleConv`) with increasing channels (32 → 64 → 128), each followed by max pooling
- A bottleneck layer with 256 filters
- Three decoder stages with transposed convolutions and skip connections, followed by `DoubleConv` blocks
- A final `1×1` convolution and sigmoid activation to output a binary road mask

The training script uses:
- **Binary Cross Entropy (BCE)** loss
- **Adam optimizer** with a learning rate of `1e-3`
- **IoU score** as the main evaluation metric
- Image size of `256×256` and a batch size of `2`

Training is performed over `10` epochs, with logging to file and console. The best model (based on validation IoU) is saved as `models/best_mini_unet.pth`.

## 📊 Results

Segmentation performance on the validation set of the DeepGlobe Road Extraction Dataset.

| Model                 | Architecture                                                                 | Training Setup                                                | Validation IoU |
|-----------------------|------------------------------------------------------------------------------|---------------------------------------------------------------|----------------|
| Mini U-Net            | Simple encoder-decoder with 3 blocks and skip connections.                   | 30 epochs · Adam (lr=1e-3) · BCE Loss                         | 0.4903         |
| Mini U-Net Plus       | Deeper version with dropout in the decoder.                                  | 30 epochs · Adam (lr=1e-3) · BCE + Dice Loss                  | 0.5153         |
| DeepLabV3 + ResNet-50 | Pretrained encoder with ASPP head, binary output layer.                      | 50 epochs · Adam (lr=1e-3) · BCE + Dice Loss                  | 0.1848         |
| Attention U-Net       | U-Net with attention gates in skip connections to focus on road structures. | 50 epochs · Adam (lr=1e-3) · ReduceLROnPlateau scheduler      | 0.5424         |
| DeepLabV3 + ResNet-101| Deeper version with COCO-pretrained encoder.                                | 50 epochs · Adam (lr=1e-4)                                    | 0.3206         |
| Segformer (HF)        | Transformer-based model pretrained for semantic segmentation.                | 50 epochs · Adam (lr=1e-4)                                    | 0.3458         |



> 📌 Metrics computed with a binary threshold of 0.5. 

## 🚀 How to run the project
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the FastAPI backend (in one terminal)
```bash
uvicorn api.main:app --reload
```
- Prediction Endpoint: http://localhost:8000/predict-image
- Swagger UI: http://localhost:8000/docs

### 3. Launch the Streamlit UI (in another terminal)
```bash
streamlit run app/ui.py
```
- Choose Upload Image to try your own files
- Choose Select from Map to download imagery from IGN WMS and run segmentation

## 🐳 Quick Start (WIP)

```bash
docker build -t road-segmentation .
docker run -p 8000:8000 road-segmentation
```

## 📊 Experiments

All training and evaluation runs are tracked using [Weights & Biases (W&B)](https://wandb.ai/), allowing full visibility into performance metrics, loss curves, and qualitative results.

📌 **Project Dashboard:**  
[🔗 Road Segmentation Project on W&B](https://api.wandb.ai/links/carmen-pastor-universidad-polit-cnica-de-madrid/nufmtfm3)

## 📁 Dataset

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/data) and place it under `data/raw/`.

---

🚧 Work in progress — baseline model, API and UI coming next.
