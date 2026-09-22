from pathlib import Path

import joblib
import numpy as np


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / "model" / "helperai_bank_scaler.pkl"


# Load the scaler used during training
scaler = joblib.load(SCALER_PATH)


def preprocess_input(features):
    """
    Preprocess customer features before sending them to the ONNX model.
    """

    features = np.array(features, dtype=np.float32).reshape(1, -1)

    # Apply the same scaling used during training
    features_scaled = scaler.transform(features)

    return features_scaled