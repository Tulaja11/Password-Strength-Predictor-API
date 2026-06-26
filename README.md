# Password Strength Predictor API

Classifies passwords as Weak, Medium, or Strong using a Random Forest trained on 670K labeled passwords — served as a FastAPI endpoint with confidence scores and per-class probabilities.

## Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 85.9% | 0.84 |
| **Random Forest (used)** | **89.5%** | **0.88** |

### The number that actually matters

89.5% accuracy sounds strong, but the per-class breakdown tells a different story:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Weak | 0.96 | **0.35** | 0.51 |
| Medium | 0.88 | 1.00 | 0.93 |
| Strong | 0.98 | 0.89 | 0.93 |

The model only catches 35% of genuinely weak passwords — `qwerty` and `123456` both get classified as Medium. The headline accuracy is propped up by the Medium class (496K of 670K rows) being easy to predict correctly. A smaller model config (75 trees, depth 15) was tested to reduce file size, but it dropped Weak recall from 0.35 to 0.14 — too much of a hit for a password-strength model, so the larger model was kept.

`/model-info` documents this limitation explicitly so anyone calling the API knows not to rely on it alone for security-critical decisions.

## A bug that needed fixing

The original notebook defined `char_tokenizer` inline. `TfidfVectorizer` pickles a reference to that function — not the code itself — so loading `vectorizer.pkl` from any other process failed:

```
AttributeError: Can't get attribute 'char_tokenizer' on <module '__main__'>
```

Fixed by moving `char_tokenizer` into `features.py`, a real importable module that both the training script and the API import from. Without this, the model trains fine but the API crashes on startup.

## How it works

- Cleans raw password text (lowercase, strip non-alpha if needed)
- Applies character-level TF-IDF — not word-level, since passwords aren't words
- Random Forest classifier with 100 trees predicts Weak / Medium / Strong
- API returns the label, confidence score, and probability for each class

## Project structure

```
password_strength_project/
├── data/
│   └── Password_Strength.csv       # 670K passwords + strength labels
├── notebooks/
│   ├── Password_Strength_Classifier.ipynb  # original notebook
│   └── 01_train_model.py           # clean training script
├── models/                         # saved model files (Git LFS)
├── features.py                     # char_tokenizer — imported by both train and API
├── api/
│   └── main.py                     # FastAPI app
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt

# Train the model
python notebooks/01_train_model.py

# Start the API
cd api
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"password": "Xk9#mPz2Qw!"}'
```

```json
{
  "password_length": 11,
  "strength_label": "Strong",
  "strength_score": 2,
  "confidence": 0.718,
  "class_probabilities": {
    "Weak": 0.026,
    "Medium": 0.256,
    "Strong": 0.718
  }
}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Predict password strength |
| GET | `/model-info` | Model stats and documented limitations |
| GET | `/health` | Health check |

## Model file size

`model.pkl` is ~48MB. Use Git LFS to track it:

```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes models/*.pkl
```

## What I'd add next

- Combine the ML prediction with rule-based checks (length, digit count, symbol presence) to catch the weak passwords the model currently misses
- Rate limiting — this would sit behind a signup form in production
- Deploy to a live URL instead of localhost only

## Dataset

Public password-strength dataset from Kaggle — ~670K passwords labeled Weak / Medium / Strong.
