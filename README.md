# HelperAI Bank — Intelligent Credit Default Prediction System
 Data → Preprocessing → DNN → Evaluation → ONNX → FastAPI → SHAP → Database → Dashboard → Docker

**HelperAI Bank** is an end-to-end Artificial Intelligence application designed to predict the likelihood of credit card default using a **Deep Neural Network (DNN)**. The system transforms customer financial and repayment information into a predictive risk assessment, provides an interpretable explanation of each prediction, stores prediction records in a database, and presents historical results through an interactive dashboard.

The project was developed to demonstrate the complete lifecycle of an AI application, from **data preprocessing and model development to model optimization, API development, explainability, database integration, dashboard visualization, and containerized deployment**.

## Project Objective

The main objective is to build an intelligent system that can analyze customer credit and repayment information and estimate whether a customer represents a potential default risk.

The system does not only provide a prediction; it also provides:

* Default probability.
* Final risk classification.
* The most influential features behind the prediction.
* Historical prediction records.
* Dashboard-level analytics.

## Machine Learning Model

The prediction model is a **Deep Neural Network (DNN)** designed for binary classification.

The model receives **23 input features**, including:

* Credit limit.
* Customer demographic information.
* Repayment status history.
* Bill amounts.
* Payment amounts.

The output is a probability between **0 and 1**, representing the estimated probability of credit default.

A decision threshold is then applied to convert the probability into a binary prediction:

* `0` → Low Default Risk
* `1` → Potential Default Risk

## Data Preprocessing

Before inference, customer data passes through the same preprocessing pipeline used during model training.

The preprocessing pipeline includes:

* Removing the customer ID because it is an identifier rather than a predictive feature.
* Separating input features from the target.
* Splitting the dataset into training and testing sets.
* Standardizing numerical features using `StandardScaler`.
* Reusing the trained scaler during inference to ensure consistency between training and deployment.

The scaler is saved as an artifact and loaded by the prediction service when the application starts.

## Model Architecture

The DNN consists of multiple fully connected layers with ReLU activation functions and a final sigmoid output layer.

The final model uses:

* Input layer: 23 features.
* Dense layer: 128 neurons.
* Dropout: 0.3.
* Dense layer: 64 neurons.
* Dropout: 0.3.
* Dense layer: 32 neurons.
* Output layer: 1 neuron with sigmoid activation.

The model was trained using:

* Adam optimizer.
* Binary Cross-Entropy loss.
* Early Stopping.
* Class weighting to improve the model's ability to identify default cases.

The final model achieved approximately:

* **Accuracy:** 82.47%
* **Precision:** 62.58%
* **Recall:** 45.89%
* **F1-Score:** 52.95%

The evaluation uses more than accuracy because the dataset contains a larger proportion of non-default cases, making metrics such as Precision, Recall, and F1-Score important for evaluating default detection.

## ONNX Deployment

After training, the DNN model was converted from Keras/TensorFlow format to **ONNX**.

The deployed prediction pipeline uses:

**ONNX Runtime**

instead of loading the original training framework for model inference.

This creates a separation between:

* Model training.
* Model deployment.
* Model inference.

The application therefore uses the exported ONNX model as the main production inference artifact.

## Explainability with SHAP

To make the predictions more interpretable, the application integrates **SHAP (SHapley Additive exPlanations)**.

For every prediction, SHAP analyzes the contribution of the input features and identifies the most influential factors.

For example, the system can return explanations such as:

* Payment Status 1 increased default risk.
* Credit Limit increased default risk.
* Payment Status 3 increased default risk.

The explanations are presented as feature contributions rather than simply returning a prediction without context.

SHAP explanations are intended to explain the model's behavior for a specific prediction; they should not be interpreted as proof of causal relationships.

## FastAPI Backend

The application exposes the AI model through a **FastAPI REST API**.

Main endpoints include:

### Health Check

```text
GET /health
```

Used to verify that the API and its main components are available.

### Prediction

```text
POST /predict
```

Receives customer information and returns:

* Prediction.
* Default probability.
* Prediction description.
* SHAP-based explanation.

### All Predictions

```text
GET /predictions
```

Returns previously stored prediction records.

### Prediction by ID

```text
GET /predictions/{prediction_id}
```

Returns the complete information associated with a specific prediction.

### Dashboard Statistics

```text
GET /dashboard/stats
```

Provides aggregated statistics for the dashboard.

### Recent Predictions

```text
GET /dashboard/recent
```

Returns the latest prediction records.

### API Documentation

```text
GET /docs
```

Provides interactive Swagger/OpenAPI documentation for testing the API.

## Database Integration

The project uses **SQLite with SQLAlchemy** to store prediction history.

Each prediction record contains information such as:

* Customer input features.
* Prediction result.
* Default probability.
* SHAP explanation.
* Prediction timestamp.

This allows the application to maintain a history of model predictions instead of treating every request as an isolated operation.

## Dashboard

The project includes a dedicated analytics dashboard that reads prediction data directly from the FastAPI backend.

The dashboard provides:

* Total number of predictions.
* Number of potential default-risk cases.
* Number of low-risk cases.
* Average default probability.
* Prediction distribution visualization.
* Recent prediction history.
* Access to individual prediction details.

The dashboard creates a simple monitoring layer over the deployed AI system and makes the stored prediction data easier to analyze.

## Docker

The entire application is containerized using **Docker**.

The Docker container packages the required application components and dependencies, including:

* FastAPI application.
* ONNX model.
* SHAP explainability.
* Database integration.
* Frontend.
* Python dependencies.

Docker Compose is used to simplify running the application locally.

The application can therefore be started as a complete service rather than manually configuring each component.

## System Architecture

```text
                    Customer Input
                         │
                         ▼
                ┌─────────────────┐
                │    FastAPI      │
                │     REST API    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Preprocessing │
                │    + Scaling    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   ONNX Runtime  │
                │    DNN Model    │
                └────────┬────────┘
                         │
                 ┌───────┴────────┐
                 ▼                ▼
          Prediction          Probability
                 │                │
                 └───────┬────────┘
                         ▼
                ┌─────────────────┐
                │      SHAP       │
                │  Explainability │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     SQLite      │
                │    Database     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Dashboard    │
                │ Analytics & Logs│
                └─────────────────┘

                         │
                         ▼
                    Docker
                         │
                         ▼
                  Deployment
```

## Technologies Used

**Programming & AI**

* Python
* TensorFlow / Keras
* Deep Neural Networks
* Scikit-learn
* NumPy

**Model Deployment**

* ONNX
* ONNX Runtime

**Explainability**

* SHAP

**Backend**

* FastAPI
* Pydantic
* Uvicorn

**Database**

* SQLite
* SQLAlchemy

**Frontend**

* HTML
* CSS
* JavaScript

**DevOps**

* Docker
* Docker Compose
* uv

## Key Features

* End-to-end DNN-based credit default prediction.
* 23-feature prediction pipeline.
* Standardized preprocessing.
* ONNX model deployment.
* SHAP-based individual prediction explanations.
* REST API using FastAPI.
* Persistent prediction history using SQLite.
* Analytics dashboard.
* Interactive API documentation.
* Dockerized application.
* Modular project architecture separating API, preprocessing, prediction, explainability, database, model, and frontend components.

## Project Structure

```text
HelperAI-Bank/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── processing.py
│   ├── prediction.py
│   └── explainability.py
│
├── database/
│   ├── __init__.py
│   ├── database.py
│   ├── models.py
│   └── init_db.py
│
├── model/
│   ├── credit_defualt_model.h5
│   ├── credit_defualt_model.onnx
│   ├── scaler.pkl
│   └── threshold.txt
│
├── frontend/
│   ├── index.html
│   └── dashboard.html
│
├── data/
│   └── helperai_bank.db
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## What This Project Demonstrates

This project demonstrates the ability to move beyond model training and build a complete AI-powered application:

**Data → Preprocessing → DNN → Evaluation → ONNX → FastAPI → SHAP → Database → Dashboard → Docker**

This makes HelperAI Bank an end-to-end AI engineering project rather than a standalone machine learning notebook.
