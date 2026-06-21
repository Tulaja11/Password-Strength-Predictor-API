"""
Shared feature extraction logic for the Password Strength Classifier.

This MUST live in its own importable module (not inline in a notebook),
because the TF-IDF vectorizer pickles a reference to `char_tokenizer` --
if that function only exists inside a notebook's __main__ namespace,
loading the vectorizer from any other process (like the API) fails with:
  AttributeError: Can't get attribute 'char_tokenizer' on <module '__main__'>

Both notebooks/01_train_model.py and api/main.py import from here so the
exact same tokenizer is used at train time and inference time.
"""


def char_tokenizer(password: str) -> list:
    """Split a password string into a list of individual characters."""
    return list(str(password))
