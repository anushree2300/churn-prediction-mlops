# Churn Prediction MLOps System
```An end-to-end MLOps pipeline for customer churn prediction using open-source tools.

Project Overview
This project implements a complete MLOps pipeline for a churn prediction system using:

Version Control: Git + GitHub
Experiment Tracking: MLflow
Model Training: Jupyter, scikit-learn
API Deployment: FastAPI
Packaging: Docker
CI/CD: GitHub Actions
Monitoring: Prometheus + Grafana
Project Structure
project-root/
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading and prediction logic
│   └── schema.py        # Pydantic models for API validation
├── notebooks/
│   └── train_model.py   # Model training script
├── models/              # Saved model artifacts
├── data/                # Dataset files
├── tests/               # Unit and integration tests
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── prometheus.yml       # Prometheus configuration
├── docker-compose.yml   # Multi-container orchestration
├── .github/workflows/   # CI/CD configuration
└── README.md            # Project documentation
Getting Started
Prerequisites
Python 3.9+
Docker and Docker Compose
Git
Installation and Setup
Clone the repository:

git clone https://github.com/anushree2300/churn-prediction-mlops.git
cd churn-prediction-mlops
Install dependencies:

pip install -r requirements.txt
Train the model:

python notebooks/train_model.py
Start the application with Docker Compose:

docker-compose up -d
Access the services:

API: http://localhost:8000
API Documentation: http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (login with admin/admin)
API Usage
Make predictions by sending POST requests to the /predict endpoint:

Bash

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0
  }'
Monitoring
The application is instrumented with Prometheus and can be monitored using Grafana dashboards. In Grafana:

Add Prometheus as a data source (URL: http://prometheus:9090)
Create dashboards for:
Request rates
Response times
Error rates
System resource usage
CI/CD Pipeline
Our GitHub Actions workflow automatically:

Runs unit tests on every push and pull request
Performs code linting with flake8
Builds and pushes Docker images when merging to main branch
Development Workflow
Create a feature branch:

git checkout -b feature/new-feature
Make changes and run tests:

pytest
Submit a pull request to the main branch```
