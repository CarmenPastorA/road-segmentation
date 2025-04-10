# Road Segmentation PoC 🚧🛰️

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
road-segmentation-poc/
├── api/                        # API logic (FastAPI/Flask)
│   ├── main.py                 # Entrypoint for serving the model
│   ├── model_loader.py         # Load model & weights
│   └── utils.py                # Pre/post-processing helpers
│
├── app/                        # UI for uploading images and showing results
│   └── ui.py                   # Gradio or Streamlit app
│
├── config/                     # Configuration files
│   └── config.yaml             # W&B, training and API configs
│
├── data/
│   ├── raw/                    # Raw dataset (from Kaggle)
│   └── processed/              # Preprocessed data
│
├── docker/
│   └── Dockerfile              # Containerization instructions
│
├── models/
│   ├── baseline_unet.py        # U-Net model
│   └── trained_model.pth       # Trained weights (not pushed to repo)
│
├── notebooks/
│   └── 01_exploration.ipynb    # Dataset exploration and visualization
│
├── scripts/
│   ├── train.py                # Training pipeline with W&B
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Run inference on a single image
│
├── shared/
│   ├── metrics.py              # IoU, Dice, etc.
│   └── transforms.py           # Image preprocessing and augmentations
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

## 🐳 Quick Start (WIP)

```bash
docker build -t road-segmentation .
docker run -p 8000:8000 road-segmentation
```

## 📊 Experiments

All experiments are tracked with [Weights & Biases](https://wandb.ai/).

## 📁 Dataset

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/data) and place it under `data/raw/`.

---

🚧 Work in progress — baseline model, API and UI coming next.
