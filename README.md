# VisaSight — US Visa Approval Prediction System

VisaSight is an end-to-end machine learning application that predicts the likely outcome of a US visa petition based on applicant, job, wage, and employer attributes. The project combines a modular ML pipeline, a FastAPI web interface, Docker containerization, and Vercel deployment.

> **Disclaimer:** VisaSight is an educational machine learning project. Its predictions are estimates based on historical data and should not be used as the sole basis for immigration, employment, or legal decisions.

## Project Overview

The project uses the EasyVisa dataset from Kaggle, containing **25,481 records**, to build a binary classification model that predicts whether a visa petition is likely to be certified or denied.

The application demonstrates the complete machine learning workflow, including data ingestion, validation, preprocessing, model selection, evaluation, and prediction serving.

- **Reported test accuracy:** Approximately 70%
- **Machine learning algorithm:** Random Forest Classifier
- **Model tuning:** GridSearchCV with 3-fold cross-validation
- **Backend:** FastAPI
- **Containerization:** Docker
- **Deployment:** Vercel

Accuracy is the currently documented evaluation metric. Further evaluation using precision, recall, F1-score, and a confusion matrix is recommended.

## Live Demo

**Application:** https://visa-sight.vercel.app/

## Key Features

- **Visa outcome prediction:** Predicts a likely petition outcome from submitted applicant and employer information.
- **Modular ML pipeline:** Separates data ingestion, validation, transformation, model training, and evaluation.
- **Random Forest classification:** Uses an ensemble learning algorithm for binary classification.
- **Hyperparameter tuning:** Uses GridSearchCV and cross-validation to select model settings.
- **Wage normalization:** Converts wage values to an annual basis to handle different pay periods.
- **FastAPI web application:** Provides a form-based interface for submitting information and viewing predictions.
- **Model artifact loading:** Loads a previously trained model for inference rather than training it for each prediction.
- **Data drift exploration:** Includes Evidently AI for examining potential data drift.
- **Docker containerization:** Packages application code and dependencies into a reproducible container image.
- **Cloud deployment:** Makes the prediction application accessible through Vercel.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | Scikit-learn, Random Forest, GridSearchCV |
| Data processing | Pandas, NumPy, SciPy |
| Backend and API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, Jinja2 |
| Data ingestion | MongoDB |
| Configuration | YAML |
| Model serialization | Pickle/serialized model artifacts, dill |
| Data drift exploration | Evidently AI |
| Containerization | Docker |
| Version control | Git, GitHub |
| Deployment | Vercel |

## Machine Learning Workflow

1. **Data ingestion:** Retrieves the training data through the project's ingestion component.
2. **Data validation:** Checks incoming data against the configured schema and validation rules.
3. **Data transformation:** Prepares features for model training and normalizes wage values.
4. **Model training:** Trains a Random Forest classifier on the prepared dataset.
5. **Hyperparameter tuning:** Uses GridSearchCV with 3-fold cross-validation to search configured model parameters.
6. **Model evaluation:** Evaluates the trained model on held-out test data. The currently documented accuracy is approximately 70%.
7. **Prediction serving:** The FastAPI application loads the saved model and generates predictions from submitted input.
8. **Containerization:** Docker packages the application and its dependencies for local execution.
9. **Deployment:** Vercel hosts the web application for online demonstration.

## Repository Structure

```text
VisaSight/
├── config/          # Model and data schema configuration
├── constants/       # Project constants
├── entity/          # Configuration and artifact entities
├── components/      # Data ingestion, validation, transformation,
│                    # model training, and evaluation
├── pipeline/        # Training and prediction pipeline orchestration
├── templates/       # Jinja2 HTML templates
├── static/          # Static frontend assets
├── saved_model/     # Saved model artifact
├── app.py           # FastAPI application entry point
├── demo.py          # Local model training workflow
├── Dockerfile       # Docker image definition
├── requirements.txt # Application dependencies
└── README.md
```

Some repository paths may change as development continues. Refer to the actual source tree for the latest structure.

## Getting Started

### Prerequisites

- Git
- Python compatible with the pinned dependencies
- Docker Desktop or Docker Engine (for containerized execution)

### 1. Clone the Repository

```bash
git clone https://github.com/PrashantByte-28/VisaSight.git
cd VisaSight
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally

```bash
python app.py
```

Open the application in your browser:

http://localhost:8080

## Run with Docker

Docker provides a consistent way to package and run VisaSight without manually installing the application dependencies in your local Python environment.

### 1. Build the Docker Image

Run this command from the project's root directory, where the `Dockerfile` is located:

```bash
docker build -t visa-app .
<<<<<<< HEAD

docker build -t visasight .
 (Readme file updated)
=======
>>>>>>> b8bfec9 (docs: improve VisaSight README and Docker instructions)
```

### 2. Start the Container

Run the container in the background and map port 8080:

```bash
<<<<<<< HEAD
docker run --rm -p 8080:8080 visa-app

docker run --rm -p 8080:8080 visasight
 (Readme file updated)
=======
docker run -d --name visa-app-container -p 8080:8080 visa-app
>>>>>>> b8bfec9 (docs: improve VisaSight README and Docker instructions)
```

### 3. Open the Application

Visit:

http://localhost:8080

Submit a prediction to check that the web application and trained model work correctly.

### 4. Verify the Container

Check whether the container is running:

```bash
docker ps
```

Check the port mapping:

```bash
docker port visa-app-container
```

### 5. View Container Logs

If the application does not load or a prediction fails, inspect the logs:

```bash
docker logs visa-app-container
```

### 6. Stop and Remove the Container

When you finish testing, stop and remove the container:

```bash
docker stop visa-app-container
docker rm visa-app-container
```

### Run in the Foreground Instead

Alternatively, run the application directly in the foreground:

```bash
docker run --rm -p 8080:8080 visa-app
```

Press `Ctrl+C` to stop it.

**Important:** The trained model artifact must exist at the path expected by the application and be included in the Docker build context. Never include API keys, database credentials, or other secrets in the Docker image or public repository.

## Model Training

Model training is separate from serving predictions. The training workflow may require additional dependencies and a configured MongoDB connection through the `MONGODB_URL` environment variable.

Install the additional training dependencies required by the project:

```bash
pip install pymongo certifi evidently imbalanced-learn neuro-mf
```

Configure the required environment variables locally before running the training workflow. Do not commit real credentials or connection strings to GitHub.

## Deployment

VisaSight is deployed on Vercel for online access. The application loads the trained model artifact to generate predictions.

Docker is provided as an additional method for packaging and running the application locally or in a compatible container environment. Docker containerization does not, by itself, mean that the container is deployed to a cloud service.

## Current Limitations and Future Improvements

- Evaluate precision, recall, F1-score, ROC-AUC or PR-AUC, and a confusion matrix.
- Add automated unit and integration tests for data processing and prediction endpoints.
- Improve validation and error handling for missing or invalid inputs.
- Introduce model versioning and reproducible experiment tracking.
- Automate data drift checks and establish an appropriate monitoring workflow.
- Assess dataset limitations, fairness, and potential bias before considering real-world use.

## Skills Demonstrated

Python · Machine Learning · Supervised Learning · Classification · Random Forest · Scikit-learn · GridSearchCV · Cross-Validation · Data Preprocessing · Feature Engineering · Model Evaluation · FastAPI · REST APIs · Docker · Containerization · MongoDB · Git · GitHub · Vercel Deployment

## Author

<<<<<<< HEAD
**Prashant Kumar Mishra**  
GitHub: https://github.com/PrashantByte-28  
LinkedIn: https://www.linkedin.com/in/prashant-kumar-mishra-356004261

LinkedIn: https://www.linkedin.com/in/prashant-kumar-mishra-356004261
 (Readme file updated)
=======
**Prashant Kumar Mishra**

- GitHub: https://github.com/PrashantByte-28
- LinkedIn: https://www.linkedin.com/in/prashant-kumar-mishra-356004261

---

*Built as a machine learning portfolio project to demonstrate model development, API integration, containerization, and deployment.*
>>>>>>> b8bfec9 (docs: improve VisaSight README and Docker instructions)
