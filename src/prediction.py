from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.processing import preprocess_input


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "credit_defualt_model.onnx"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.txt"


# Load ONNX model
session = ort.InferenceSession(str(MODEL_PATH))

input_name = session.get_inputs()[0].name


# Load threshold
with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())


def predict(features):
    """
    Predict whether a customer is likely to default.
    """

    # Preprocess input
    features_scaled = preprocess_input(features)

    # ONNX prediction
    output = session.run(
        None,
        {input_name: features_scaled}
    )

    # Get probability
    y_prob = float(np.asarray(output[0]).ravel()[0])

    # Apply threshold
    y_pred = int(y_prob >= threshold)

    return {
        "probability": y_prob,
        "prediction": y_pred
    }