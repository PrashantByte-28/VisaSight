# VisaSight — US Visa Approval Prediction System

**VisaSight** is a production-oriented machine learning application that predicts the likely outcome of a US work-visa petition from applicant, job, wage, and employer attributes. It combines a modular machine learning pipeline with a FastAPI web interface and a live Vercel deployment.

> **Important:** VisaSight is an educational decision-support demo, not an official immigration tool. Its predictions are estimates and must not be used as the sole basis for immigration, employment, or legal decisions.

## Project Overview

The goal of VisaSight is to demonstrate an end-to-end machine learning workflow for a real-world classification problem: estimating whether a US visa petition is likely to be certified or denied based on patterns in historical data. The project covers data processing, model selection, evaluation, artifact management, and serving predictions through a web application.

The model was developed using the EasyVisa dataset from Kaggle (25,481 records). The reported held-out test accuracy is approximately **70%**. Accuracy is the currently reported evaluation metric; additional metrics such as precision, recall, F1-score, and a confusion matrix should be evaluated before drawing stronger conclusions about model performance.

## Key Features

- **Binary classification:** predicts a likely visa-petition outcome from submitted features.
- **Modular ML pipeline:** separates data ingestion, validation, transformation, model training, and evaluation.
- **Model selection:** uses scikit-learn's `GridSearchCV` with cross-validation to tune a Random Forest classifier.
- **Wage normalization:** converts wage values to an annual basis to handle records expressed using different pay periods.
- **Prediction web app:** provides a form-based interface built with FastAPI and Jinja2.
- **Model artifact loading:** serves predictions using a previously trained model artifact; model training is performed locally, not during a Vercel request.
- **Data-drift exploration:** includes Evidently AI as part of the project's monitoring work.
- **Cloud deployment:** the prediction application is deployed on Vercel.

## Technical Skills Demonstrated

Python · Machine Learning · Supervised Learning · Classification · Random Forest · Scikit-learn · GridSearchCV · Cross-Validation · Data Preprocessing · Feature Transformation · Model Evaluation · Pipeline Design · FastAPI · REST API Concepts · Jinja2 · MongoDB (training-data ingestion) · Evidently AI · Git · GitHub · Vercel Deployment

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 for deployment |
| Machine learning | scikit-learn, Random Forest, GridSearchCV |
| Data processing | pandas, NumPy, SciPy |
| API and interface | FastAPI, Uvicorn, Jinja2 |
| Configuration and artifacts | YAML, dill, serialized model artifact |
| Training data ingestion | MongoDB |
| Monitoring exploration | Evidently AI |
| Deployment | Vercel, GitHub |

## Machine Learning Workflow

1. **Data ingestion:** retrieves the training data through the project's ingestion component.
2. **Data validation:** checks the incoming dataset against the project's schema and validation rules.
3. **Data transformation:** prepares model features and normalizes wage values to an annual basis.
4. **Model training and selection:** trains candidate configurations and tunes the configured Random Forest model using `GridSearchCV` with 3-fold cross-validation.
5. **Evaluation:** evaluates the selected model on held-out test data. The currently documented test accuracy is approximately 70%.
6. **Prediction serving:** the FastAPI application loads the saved model artifact and returns a prediction from the submitted form.
7. **Monitoring exploration:** Evidently AI is included for examining potential data drift; scheduled production drift monitoring is not currently claimed.

## Repository Structure

The project follows a modular structure:

```text
constants/   # Fixed values and configuration
entity/      # Configuration and artifact entities
components/  # Ingestion, validation, transformation, training, evaluation
pipeline/    # Training and prediction pipeline orchestration
config/      # Model and data-schema configuration
templates/   # Jinja2 HTML templates
static/      # Frontend assets
saved_model/ # Serialized model artifact used by the app
app.py       # FastAPI application entry point
demo.py      # Local training workflow entry point
```

Some paths may differ as the repository evolves; use the actual source tree as the source of truth.

## Live Demo

**Application:** https://visa-sight.vercel.app/

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/PrashantByte-28/VisaSight.git
cd VisaSight
```

### 2. Install application dependencies

Use Python compatible with the pinned dependencies, then run:

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Open http://localhost:8080 in your browser.

### Training the model

Training is separate from the deployed prediction application and requires additional training dependencies and a MongoDB connection configured through `MONGODB_URL`. Do not commit credentials or connection strings to Git.

```bash
pip install pymongo certifi evidently imbalanced-learn neuro-mf
```

Set `MONGODB_URL` in your local environment before running the training entry point. Never place real secrets directly in source code or public repository files.

## Deployment

The prediction application is hosted on Vercel. The deployment serves the FastAPI app and loads the saved model artifact. Training is performed locally and is not run as part of an incoming prediction request.

## Current Limitations and Next Steps

- Evaluate precision, recall, F1-score, ROC-AUC or PR-AUC, and a confusion matrix in addition to accuracy.
- Add unit and integration tests for data validation, transformation, model loading, and prediction endpoints.
- Validate input ranges and handle invalid or missing form values consistently.
- Add reproducible experiment tracking and document model versions.
- Automate data-drift checks and define an appropriate monitoring process.
- Review fairness, dataset limitations, and potential bias before any real-world use.

## Author

**Prashant Kumar Mishra**  
GitHub: https://github.com/PrashantByte-28  
LinkedIn: https://www.linkedin.com/in/prashant-kumar-mishra-356004261
