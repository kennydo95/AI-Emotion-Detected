"""
predict.py — Load the saved model and predict a single message.
"""

import joblib

MODEL_PATH = "models/spam_model.pkl"
_pipeline = None


def load_model():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict(message: str) -> dict:
    """
    Returns:
        {
            "label":      "Spam" or "Ham",
            "confidence": 0.0 – 1.0,
            "message":    original message
        }
    """
    pipeline = load_model()
    prob = pipeline.predict_proba([message])[0]   # [ham_prob, spam_prob]
    label_idx = int(prob.argmax())
    return {
        "label":      "Spam" if label_idx == 1 else "Ham",
        "confidence": round(float(prob[label_idx]), 4),
        "message":    message,
    }


if __name__ == "__main__":
    tests = [
        "Congratulations! You've won a FREE iPhone. Click here now!!!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your bank account has been suspended. Verify now.",
        "Can you send me the notes from today's class?",
    ]
    for t in tests:
        result = predict(t)
        icon = "🚨" if result["label"] == "Spam" else "✅"
        print(f"{icon} [{result['label']} {result['confidence']*100:.1f}%] {t[:60]}")
