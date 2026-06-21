"""
Password Strength Predictor — FastAPI service.

Loads a Random Forest classifier trained on character-level TF-IDF features
to predict whether a password is Weak / Medium / Strong.

Run locally:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from features import char_tokenizer  # noqa: F401 -- required for vectorizer.pkl to unpickle correctly

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Password Strength Predictor",
    description="Predicts whether a password is Weak, Medium, or Strong using "
                "character-level TF-IDF features and a Random Forest classifier "
                "trained on ~670K labeled passwords.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

LABELS = {0: "Weak", 1: "Medium", 2: "Strong"}


class PasswordInput(BaseModel):
    password: str = Field(..., min_length=1, max_length=128, description="Password to evaluate")

    class Config:
        json_schema_extra = {"example": {"password": "Xk9#mPz2Qw!"}}


class PredictionResponse(BaseModel):
    password_length: int
    strength_label: str
    strength_score: int
    confidence: float
    class_probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": metadata["model_name"],
        "accuracy": round(metadata["accuracy"], 3),
        "f1_weighted": round(metadata["f1_weighted"], 3),
        "labels": LABELS,
        "note": (
            "Weak-class recall is lower (~0.35) than overall accuracy suggests -- "
            "the model is biased toward predicting Medium, so some genuinely weak "
            "passwords get misclassified. Don't rely on this alone for security-critical "
            "gating; combine with rule-based checks (length, character variety) for production use."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PasswordInput):
    try:
        vec = vectorizer.transform([payload.password])
        pred = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]

        return PredictionResponse(
            password_length=len(payload.password),
            strength_label=LABELS[pred],
            strength_score=pred,
            confidence=round(float(proba[pred]), 3),
            class_probabilities={LABELS[i]: round(float(p), 3) for i, p in enumerate(proba)},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
