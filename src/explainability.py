from pathlib import Path

import joblib
import numpy as np
import shap
import tensorflow as tf


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "credit_defualt_model.h5"
SCALER_PATH = BASE_DIR / "model" / "helperai_bank_scaler.pkl"


# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Feature names used by the model
FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# Create SHAP explainer
background_data = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)

explainer = shap.GradientExplainer(
    model,
    background_data
)


def explain_prediction(features, top_n=5):
    """
    Explain the most important features for one prediction.
    """

    # Convert input to numpy array
    features = np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    # Apply the same scaler used during training
    features_scaled = scaler.transform(features)

    # Calculate SHAP values
    shap_values = explainer.shap_values(features_scaled)

    shap_values = np.asarray(shap_values)

    # Handle different SHAP output shapes
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    # Sort features by absolute SHAP value
    important_indices = np.argsort(
        np.abs(shap_values)
    )[::-1][:top_n]

    reasons = []

    for index in important_indices:

        contribution = float(shap_values[index])

        if contribution > 0:
            effect = "increases default risk"
        else:
            effect = "decreases default risk"

        reasons.append({
            "feature": FEATURE_NAMES[index],
            "shap_value": round(contribution, 4),
            "effect": effect
        })

    return reasons