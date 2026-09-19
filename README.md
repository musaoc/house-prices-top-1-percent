# House Prices Prediction — Kaggle Top 1% Solution

An end-to-end machine learning project that predicts residential home prices using comprehensive feature engineering, statistical outlier treatment, and tuned gradient boosting models to achieve a Top 1% ranking on Kaggle.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/top-1-housing-price-eda-random-for-everyone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Tabular%20Regression%20/%20Ensembling-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[House Prices Prediction — Kaggle Top 1% Solution](https://www.kaggle.com/code/lazer999/top-1-housing-price-eda-random-for-everyone)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Achieved **Top 1% ranking** on the global Kaggle leaderboard among thousands of competing models.
- Rigorous exploratory data analysis (EDA) identifying critical price drivers (living area, overall quality, neighborhood).
- Robust missing value handling using domain-specific heuristics (e.g., distinguishing absent amenities vs missing data).
- Benchmarked Random Forest Regressor and Gradient Boosting Regressors with cross-validation.
- Leak-free feature encoding with Scikit-Learn `ColumnTransformer` and pipeline components.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[Raw Housing Dataset] --> B[Exploratory Data Analysis]
    B --> C[Outlier Treatment & Missing Imputation]
    C --> D[One-Hot & Ordinal Feature Encoding]
    D --> E[Gradient Boosting / RF Regressor]
    E --> F[Cross-Validation Optimization]
    F --> G[Top 1% Predictions]
```

---

## Repository Structure

```plaintext
house-prices-top-1-percent/
├── notebooks/
│   └── house-prices-top-1-percent.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/house-prices-top-1-percent.git
cd house-prices-top-1-percent
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/house-prices-top-1-percent.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/home-data-for-ml-course)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle competitions download -c home-data-for-ml-course
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [House Prices Prediction — Kaggle Top 1% Solution](https://www.kaggle.com/code/lazer999/top-1-housing-price-eda-random-for-everyone)

If you found this project helpful or insightful, please consider starring the repository ⭐!
