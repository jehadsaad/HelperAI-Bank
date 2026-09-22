from pathlib import Path
import json

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.database import Base, engine, get_db
from database.models import Prediction
from src.explainability import explain_prediction
from src.prediction import predict


# ============================================================
# Database Initialization
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# Application Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_PATH = BASE_DIR / "fronted" / "index.html"
DASHBOARD_PATH = BASE_DIR / "fronted" / "dashboard.html"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="HelperAI Bank API",
    description=(
        "Credit Card Default Prediction API "
        "using Deep Neural Network, ONNX Runtime, "
        "SHAP Explainability, SQLite Database, "
        "and Docker."
    ),
    version="1.0.0",
)


# ============================================================
# Customer Input Schema
# ============================================================

class CustomerData(BaseModel):

    # Customer information
    LIMIT_BAL: float = Field(
        ...,
        ge=0,
        description="Credit limit"
    )

    SEX: int = Field(
        ...,
        description="Sex: 1 = Male, 2 = Female"
    )

    EDUCATION: int = Field(
        ...,
        description="Education level"
    )

    MARRIAGE: int = Field(
        ...,
        description="Marital status"
    )

    AGE: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer age"
    )

    # Repayment status
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    # Bill amounts
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    # Payment amounts
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# ============================================================
# Convert Customer Data to Model Features
# ============================================================

def customer_to_features(
    customer: CustomerData
) -> list[float]:

    return [
        customer.LIMIT_BAL,
        customer.SEX,
        customer.EDUCATION,
        customer.MARRIAGE,
        customer.AGE,

        customer.PAY_0,
        customer.PAY_2,
        customer.PAY_3,
        customer.PAY_4,
        customer.PAY_5,
        customer.PAY_6,

        customer.BILL_AMT1,
        customer.BILL_AMT2,
        customer.BILL_AMT3,
        customer.BILL_AMT4,
        customer.BILL_AMT5,
        customer.BILL_AMT6,

        customer.PAY_AMT1,
        customer.PAY_AMT2,
        customer.PAY_AMT3,
        customer.PAY_AMT4,
        customer.PAY_AMT5,
        customer.PAY_AMT6,
    ]


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "HelperAI Bank API",
        "version": "1.0.0",
        "database": "SQLite",
        "model": "ONNX",
        "explainability": "SHAP",
    }


# ============================================================
# Home Page
# ============================================================

@app.get("/")
def home():

    if not FRONTEND_PATH.exists():

        raise HTTPException(
            status_code=500,
            detail="Frontend file not found."
        )

    return FileResponse(FRONTEND_PATH)


# ============================================================
# Dashboard Page
# ============================================================

@app.get("/dashboard")
def dashboard():

    if not DASHBOARD_PATH.exists():

        raise HTTPException(
            status_code=500,
            detail="Dashboard file not found."
        )

    return FileResponse(DASHBOARD_PATH)


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def make_prediction(
    customer: CustomerData,
    db: Session = Depends(get_db)
):

    try:

        # Convert input to model features
        features = customer_to_features(customer)

        # Run ONNX prediction
        result = predict(features)

        # Generate SHAP explanation
        reasons = explain_prediction(
            features,
            top_n=5
        )

        # Convert prediction to readable text
        if result["prediction"] == 1:

            prediction_text = "Potential Default Risk"

        else:

            prediction_text = "Low Default Risk"


        # ====================================================
        # Save Prediction to Database
        # ====================================================

        prediction_record = Prediction(

            # Customer information
            limit_bal=customer.LIMIT_BAL,
            sex=customer.SEX,
            education=customer.EDUCATION,
            marriage=customer.MARRIAGE,
            age=customer.AGE,

            # Repayment status
            pay_0=customer.PAY_0,
            pay_2=customer.PAY_2,
            pay_3=customer.PAY_3,
            pay_4=customer.PAY_4,
            pay_5=customer.PAY_5,
            pay_6=customer.PAY_6,

            # Bill amounts
            bill_amt1=customer.BILL_AMT1,
            bill_amt2=customer.BILL_AMT2,
            bill_amt3=customer.BILL_AMT3,
            bill_amt4=customer.BILL_AMT4,
            bill_amt5=customer.BILL_AMT5,
            bill_amt6=customer.BILL_AMT6,

            # Payment amounts
            pay_amt1=customer.PAY_AMT1,
            pay_amt2=customer.PAY_AMT2,
            pay_amt3=customer.PAY_AMT3,
            pay_amt4=customer.PAY_AMT4,
            pay_amt5=customer.PAY_AMT5,
            pay_amt6=customer.PAY_AMT6,

            # Prediction
            prediction=result["prediction"],
            default_probability=result["probability"],

            # SHAP reasons
            reasons=json.dumps(reasons),
        )


        # Save record
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)


        # ====================================================
        # API Response
        # ====================================================

        return {

            "id": prediction_record.id,

            "prediction": result["prediction"],

            "prediction_text": prediction_text,

            "default_probability": round(
                result["probability"],
                4
            ),

            "reasons": reasons,
        }


    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# Get All Predictions
# ============================================================

@app.get("/predictions")
def get_predictions(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(Prediction)
            .order_by(Prediction.id.desc())
            .all()
        )

        results = []

        for record in records:

            results.append({

                "id": record.id,

                "prediction": record.prediction,

                "default_probability": round(
                    record.default_probability,
                    4
                ),

                "created_at": record.created_at,
            })


        return {

            "count": len(results),

            "predictions": results
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve predictions: "
                f"{str(error)}"
            )
        )


# ============================================================
# Get Prediction by ID
# ============================================================

@app.get("/predictions/{prediction_id}")
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    record = (
        db.query(Prediction)
        .filter(
            Prediction.id == prediction_id
        )
        .first()
    )


    if record is None:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )


    # Convert stored SHAP JSON back to Python object
    try:

        reasons = json.loads(
            record.reasons
        )

    except (
        TypeError,
        json.JSONDecodeError
    ):

        reasons = []


    return {

        "id": record.id,

        "customer": {

            "LIMIT_BAL": record.limit_bal,
            "SEX": record.sex,
            "EDUCATION": record.education,
            "MARRIAGE": record.marriage,
            "AGE": record.age,

            "PAY_0": record.pay_0,
            "PAY_2": record.pay_2,
            "PAY_3": record.pay_3,
            "PAY_4": record.pay_4,
            "PAY_5": record.pay_5,
            "PAY_6": record.pay_6,

            "BILL_AMT1": record.bill_amt1,
            "BILL_AMT2": record.bill_amt2,
            "BILL_AMT3": record.bill_amt3,
            "BILL_AMT4": record.bill_amt4,
            "BILL_AMT5": record.bill_amt5,
            "BILL_AMT6": record.bill_amt6,

            "PAY_AMT1": record.pay_amt1,
            "PAY_AMT2": record.pay_amt2,
            "PAY_AMT3": record.pay_amt3,
            "PAY_AMT4": record.pay_amt4,
            "PAY_AMT5": record.pay_amt5,
            "PAY_AMT6": record.pay_amt6,
        },

        "prediction": record.prediction,

        "prediction_text": (
            "Potential Default Risk"
            if record.prediction == 1
            else "Low Default Risk"
        ),

        "default_probability": round(
            record.default_probability,
            4
        ),

        "reasons": reasons,

        "created_at": record.created_at,
    }


# ============================================================
# Dashboard Statistics
# ============================================================

@app.get("/dashboard/stats")
def dashboard_stats(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(Prediction)
            .all()
        )


        # Total predictions
        total_predictions = len(records)


        # Potential default risk
        default_risk = sum(
            1
            for record in records
            if record.prediction == 1
        )


        # Low risk
        low_risk = sum(
            1
            for record in records
            if record.prediction == 0
        )


        # Average probability
        if total_predictions > 0:

            average_probability = sum(
                record.default_probability
                for record in records
            ) / total_predictions

        else:

            average_probability = 0.0


        return {

            "total_predictions": total_predictions,

            "default_risk": default_risk,

            "low_risk": low_risk,

            "average_probability": round(
                average_probability,
                4
            ),
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load dashboard statistics: "
                f"{str(error)}"
            )
        )


# ============================================================
# Dashboard Recent Predictions
# ============================================================

@app.get("/dashboard/recent")
def dashboard_recent_predictions(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(Prediction)
            .order_by(
                Prediction.id.desc()
            )
            .limit(10)
            .all()
        )


        results = []


        for record in records:

            results.append({

                "id": record.id,

                "prediction": record.prediction,

                "prediction_text": (
                    "Potential Default Risk"
                    if record.prediction == 1
                    else "Low Default Risk"
                ),

                "default_probability": round(
                    record.default_probability,
                    4
                ),

                "created_at": record.created_at,
            })


        return {

            "count": len(results),

            "predictions": results
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load recent predictions: "
                f"{str(error)}"
            )
        )


# ============================================================
# API Information
# ============================================================

@app.get("/api-info")
def api_info():

    return {

        "project": "HelperAI Bank",

        "purpose": (
            "Credit Card Default Prediction"
        ),

        "model": (
            "Deep Neural Network"
        ),

        "inference_engine": (
            "ONNX Runtime"
        ),

        "explainability": "SHAP",

        "database": "SQLite",

        "dashboard": True,

        "endpoints": {

            "home": "/",

            "health": "/health",

            "api_info": "/api-info",

            "prediction": "POST /predict",

            "all_predictions": (
                "GET /predictions"
            ),

            "prediction_by_id": (
                "GET /predictions/{prediction_id}"
            ),

            "dashboard": "/dashboard",

            "dashboard_stats": (
                "GET /dashboard/stats"
            ),

            "dashboard_recent": (
                "GET /dashboard/recent"
            ),

            "documentation": "/docs",
        },
    }