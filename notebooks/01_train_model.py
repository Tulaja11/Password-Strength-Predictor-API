"""
Train the password strength classifier and save model + vectorizer.

This is a script version of the original notebook's logic, with one
critical fix: char_tokenizer is imported from features.py (a real module)
instead of being defined inline, so the saved vectorizer.pkl can be
loaded correctly from any other process (like the FastAPI app).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from features import char_tokenizer

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Password_Strength.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
print("Raw shape:", df.shape)

df = df.dropna()
df = df.drop_duplicates(subset='password')
print("Shape after cleaning:", df.shape)

X = df['password'].astype(str)
y = df['strength']

vectorizer = TfidfVectorizer(tokenizer=char_tokenizer, lowercase=False)
X_vec = vectorizer.fit_transform(X)
print("Vectorized shape:", X_vec.shape, "| Vocab size:", len(vectorizer.vocabulary_))

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("F1 (weighted):", f1_score(y_test, lr_pred, average='weighted'))

print("\n--- Random Forest ---")
# Original config (100 trees, depth 20). A leaner version (75/15) was tested to
# shrink the ~48MB pickle, but it dropped Weak-class recall from 0.35 to 0.14 --
# too large a hit for a password-strength model, where catching weak passwords
# is the whole point. Keeping the stronger model and handling file size via
# Git LFS instead (see README) rather than trading away the model's one job.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print("Accuracy:", rf_acc)
print("F1 (weighted):", rf_f1)
print("\n", classification_report(y_test, rf_pred, target_names=['Weak', 'Medium', 'Strong']))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
joblib.dump({'model_name': 'Random Forest', 'accuracy': rf_acc, 'f1_weighted': rf_f1},
            os.path.join(MODEL_DIR, 'metadata.pkl'))

print("\nSaved model.pkl, vectorizer.pkl, metadata.pkl to models/")
