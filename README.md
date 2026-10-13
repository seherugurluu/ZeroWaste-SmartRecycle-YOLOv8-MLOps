# ZeroWaste-SmartRecycle-YOLOv8-MLOps
# Smart Zero Waste and Recycling Management System Based on YOLOv8 ♻️

## 1. Project Overview

This project aims to develop a smart waste classification and recycling management system using YOLOv8 and MLOps principles.

The system is designed to identify different types of waste from images and provide appropriate recycling guidance based on the detected category. It also aims to support waste management awareness through a Zero Waste Score.

## 2. Project Objectives

- Detect and classify waste using YOLOv8.
- Support six waste categories.
- Provide recycling guidance based on the detected waste type.
- Develop a reproducible machine learning workflow.
- Apply MLOps practices for data versioning, model development, testing, deployment, and monitoring.

## 3. Waste Categories

The dataset includes the following categories:

- Biodegradable
- Cardboard
- Glass
- Metal
- Paper
- Plastic

## 4. Technologies

- **Python:** Main programming language
- **YOLOv8 (Ultralytics):** Object detection
- **OpenCV:** Image processing
- **Poetry:** Dependency management
- **DVC:** Dataset versioning
- **Git and GitHub:** Version control and collaboration
- **Docker:** Containerization

Additional MLOps tools will be integrated as the project develops.

## Training and Evaluation

Training parameters are stored in [configs/train.yaml](configs/train.yaml).
Start a reproducible YOLOv8 detection training run with:

```powershell
poetry run python src/train.py --config configs/train.yaml
```

After training, evaluate the best checkpoint and create the metric files:

```powershell
poetry run python src/evaluate.py --model runs/train/waste-yolov8/weights/best.pt
```

The evaluation command writes `reports/metrics.json` and
`reports/evaluation_report.md`. Metrics must be generated from a compatible
detection model; the old `models/best.pt` file is a classification model and
cannot be evaluated against the current `images/labels` detection dataset.

## Docker

Docker Desktop must be running before using the commands below. The dataset is
mounted read-only into the container and is not copied into the image.

Validate the dataset:

```powershell
docker compose run --rm dataset-validator
```

Run the automated dataset tests:

```powershell
docker compose run --rm dataset-tests
```

The current Docker setup validates the dataset. A model API container will be
added after the FastAPI serving layer is implemented.

## CI/CD

GitHub Actions workflows are stored in `.github/workflows/`.

- `ci.yml` runs on pushes and pull requests to `main`/`master`. It runs
  pre-commit, pytest, Docker Compose configuration validation, and a Docker
  image build.
- `cd.yml` runs for version tags such as `v1.0.0` and publishes the validator
  image to GitHub Container Registry (GHCR).

To publish a release image, create and push a version tag:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

The GitHub Actions `GITHUB_TOKEN` is used for GHCR authentication; no personal
access token is stored in the repository.

## 5. Project Structure

```text
ZeroWaste-SmartRecycle-YOLOv8-MLOps/
├── configs/
├── data/
├── models/
├── notebooks/
├── src/
│   └── validate_dataset.py
├── tests/
├── .dvc/
├── .dvcignore
├── .gitignore
├── data.dvc
├── pyproject.toml
├── poetry.lock
└── README.md
```

## 6. Dataset

The project uses the [Garbage Detection – 6 Waste Categories dataset](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection).

The dataset is organized into training, validation, and test splits. Dataset versioning is managed using DVC.

The dataset files are not stored directly in the GitHub repository.

## 7. Team

- Seher Uğurlu
- Damla Su Özbek

**Course:** MLOps

