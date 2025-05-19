# Road Segmentation from Satellite Images 🚧🛰️

This project is an end-to-end solution for segmenting roads from satellite images using the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset). It includes a segmentation model, a FastAPI backend, a Streamlit UI, and a CI/CD pipeline for deployment on Azure.

## 🔍 Objective

- Develop and train segmentation models for road extraction.
- Serve predictions via a REST API using FastAPI.
- Provide a user-friendly UI with Streamlit for image uploads and visualization.
- Track experiments with Weights & Biases (W&B).
- Containerize the application with Docker.
- Deploy to Azure using Azure Container Registry (ACR) and App Services.

## 📂 Project Structure

```
road-segmentation/
├── .github/workflows/          # CI/CD pipeline
│   └── deploy.yml              # GitHub Actions workflow
├── api/                        # FastAPI backend
│   ├── main.py                 # API entrypoint
│   ├── model_loader.py         # Model loading logic
│   ├── utils.py                # Image pre/post-processing
│   └── .gitkeep                # Placeholder
├── app/                        # Streamlit UI
│   ├── api_client.py           # API communication for UI
│   ├── Dockerfile              # UI container
│   ├── ui.py                   # Streamlit application
│   └── .gitkeep                # Placeholder
├── config/                     # Configuration files
│   └── .gitkeep                # Placeholder
├── data/                       # Dataset and preprocessing
│   ├── dataset.py              # PyTorch Dataset class
│   ├── example_dataloader.py   # Dataloader preview script
│   ├── __init__.py             # Package initialization
├── deployment/                 # Deployment scripts
│   ├── Dockerfile              # Alternative deployment container
│   ├── startup.sh              # Startup script for containers
├── docker/                     # Docker configurations
│   └── .gitkeep                # Placeholder
├── models/                     # Model definitions and weights
│   ├── mini_unet.py            # Mini U-Net implementation
│   ├── unet_variants.py        # U-Net and variants (e.g., Attention U-Net)
│   ├── __init__.py             # Package initialization
│   └── .gitkeep                # Placeholder
├── notebooks/                  # Exploratory notebooks
│   └── .gitkeep                # Placeholder
├── scripts/                    # Training and evaluation scripts
│   ├── create_subset.py        # Creates a smaller training set
│   ├── evaluate.py             # Model evaluation
│   ├── evaluate_models_wandb.py # W&B evaluation script
│   ├── predict.py              # Single-image inference
│   ├── train.py                # Model training with W&B
│   └── .gitkeep                # Placeholder
├── shared/                     # Shared utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── transforms.py           # Albumentations transforms
│   └── .gitkeep                # Placeholder
├── tests/                      # Unit tests
│   ├── empty.jpg               # Empty image for testing
│   ├── invalid.txt             # Invalid file for testing
│   ├── test.jpg                # Original test image
│   ├── test.py                 # Test script
│   ├── test_api.py             # API endpoint tests
│   └── .gitkeep                # Placeholder
├── .dockerignore               # Docker ignore rules
├── .gitignore                  # Git ignore rules
├── docker-compose.yml          # Local multi-container setup
├── Dockerfile.api              # API container
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## 🧠 Model Architectures & Training

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

## 🚀 Getting Started

### 🖥️ Local Development Environment

For users who want to run the application on their local machine.

#### Prerequisites
- Python 3.10
- Docker and Docker Compose (optional, for containerized setup)
- [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) (place in `data/raw/`)

#### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/road-segmentation.git
   cd road-segmentation
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt -i https://pypi.org/simple -i https://download.pytorch.org/whl/cpu
   ```

4. **Ensure model weights are available**:
   - Verify that `models/best_attention.pth` exists.
   - Generate a lightweight model:
     ```bash
     python -c "import torch; from models.unet_variants import get_unet_variant; model = get_unet_variant('attunet', in_channels=3, out_channels=1); torch.save(model.state_dict(), 'models/best_attention.pth')"
     ```

#### Running the Application
1. **Start the FastAPI backend**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Access the API at `http://localhost:8000/predict-image`.
   - Explore endpoints at `http://localhost:8000/docs` (Swagger UI).

2. **Launch the Streamlit UI** (in a separate terminal):
   ```bash
   streamlit run app/ui.py
   ```
   - Open `http://localhost:7860` in your browser.
   - Upload an image or select imagery from IGN WMS for segmentation.

3. **(Optional) Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   - API: `http://localhost:8000`
   - UI: `http://localhost:7860`

### ☁️ Production Deployment on Azure

For users who want to deploy the service using their Azure account.

#### Prerequisites
- Azure account with an active subscription.
- Azure CLI installed (`az` command).
- GitHub repository with the project.
- Azure Container Registry (ACR) and two App Services (`roadseg-api`, `roadseg-ui`).

#### Setup
1. **Create Azure Resources**:
   - **Azure Container Registry**:
     ```bash
     az acr create --resource-group road-mlops-rg --name roadsegacr --sku Basic
     ```
   - **App Services**:
     ```bash
     az appservice plan create --name myAppServicePlan --resource-group road-mlops-rg --sku B1
     az webapp create --resource-group road-mlops-rg --plan myAppServicePlan --name roadseg-api --deployment-container-image-name roadsegacr.azurecr.io/road-segmentation-api:latest
     az webapp create --resource-group road-mlops-rg --plan myAppServicePlan --name roadseg-ui --deployment-container-image-name roadsegacr.azurecr.io/road-segmentation-ui:latest
     ```

2. **Configure GitHub Secrets**:
   - Obtain ACR credentials:
     ```bash
     az acr credential show --name roadsegacr
     ```
     - Add `ACR_USERNAME` and `ACR_PASSWORD` to GitHub Secrets.
   - Obtain App Service publishing profiles:
     - From Azure Portal: App Services → `roadseg-api`/`roadseg-ui` → Download publish profile.
     - Add `AZURE_WEBAPP_PUBLISH_PROFILE_API` and `AZURE_WEBAPP_PUBLISH_PROFILE_UI` (full XML content) to GitHub Secrets.
   - If using Azure Blob Storage for `best_attention.pth`:
     - Add `AZURE_STORAGE_CONNECTION_STRING` to GitHub Secrets.

3. **Configure CI/CD**:
   - Ensure `.github/workflows/deploy.yml` is configured.
   - Push changes to trigger the pipeline:
     ```bash
     git add .
     git commit -m "Update for Azure deployment"
     git push origin main
     ```


#### Accessing the Deployed Service
- **API**: `https://roadseg-api.azurewebsites.net/predict-image`
- **UI**: `https://roadseg-ui.azurewebsites.net`
- Verify the services are running in Azure Portal → App Services → Overview.

## 🌐 Production URLs

The application is deployed on Azure and accessible at:

- **API**: [https://roadseg-api.azurewebsites.net](https://roadseg-api.azurewebsites.net)
  - Endpoint: `/predict-image` (POST, accepts image files)
  - Swagger UI: `/docs`
- **UI**: [https://roadseg-ui.azurewebsites.net](https://roadseg-ui.azurewebsites.net)
  - Upload images or select IGN WMS imagery for road segmentation.

> 📌 Ensure the App Services are running and WebSockets are enabled for the UI. Check logs in Azure Portal if the URLs are inaccessible.

## 📊 Experiment Tracking

Experiments are tracked using [Weights & Biases (W&B)](https://wandb.ai/) for metrics, loss curves, and visualizations.

📌 **Project Dashboard**:  
[🔗 Road Segmentation Project on W&B](https://api.wandb.ai/links/carmen-pastor-universidad-polit-cnica-de-madrid/nufmtfm3)

## 📁 Dataset

Download the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)

## 📚 References

- [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Weights & Biases](https://wandb.ai/)
- [Azure App Services](https://learn.microsoft.com/en-us/azure/app-service/)

---

This project was developed by Carmen Pastor ([@CarmenPastorA](https://github.com/CarmenPastorA)) and Lorena Sánchez ([@LorenaSanchezC](https://github.com/LorenaSanchezC)).

---

🚀 **Contributions welcome!** Open issues or PRs to improve the project.