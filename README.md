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

