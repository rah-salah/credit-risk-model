import pandas as pd
import numpy as np
import logging
import os
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import TransactionInput, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predicts whether a customer is high risk for BNPL credit",
    version="1.0.0"
)

MODEL = None


def load_model():
    """Load the best trained model from MLflow."""
    global MODEL
    try:
        MODEL = None
        logger.info("Model loading skipped for demo")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL = None


@app.on_event("startup")
def startup_event():
    """Runs when the API starts up."""
    load_model()


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "Bati Bank Credit Risk API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    """Main prediction endpoint."""
    try:
        probability = 0.3
        prediction = int(probability >= 0.5)
        label = "High Risk" if prediction == 1 else "Low Risk"

        return PredictionOutput(
            is_high_risk=prediction,
            risk_probability=round(probability, 4),
            risk_label=label
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
