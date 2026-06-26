# Password Strength Predictor API

Trained a Random Forest on 670K labeled passwords using character-level TF-IDF features to classify passwords as Weak, Medium, or Strong. Wrapped it in a FastAPI endpoint that returns the predicted label, confidence score, and probabilities for all three classes.

## Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 85.9% | 0.84 |
| Random Forest | 89.5% | 0.88 |

Random Forest was used as the final model. One thing worth noting — it only catches 35% of genuinely weak passwords (Weak-class recall = 0.35). `qwerty` and `123456` both get classified as Medium. The `/model-info` endpoint documents this.

## Stack

Python, Scikit-learn, FastAPI, Pydantic, joblib

## Run it

```bash
pip install -r requirements.txt
python notebooks/01_train_model.py
cd api && uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/predict` | Returns strength label + confidence |
| GET | `/model-info` | Model stats and known limitations |
| GET | `/health` | Health check |

## Dataset

Kaggle — ~670K passwords labeled Weak / Medium / Strong.
