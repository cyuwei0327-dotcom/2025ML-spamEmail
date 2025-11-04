import joblib
import argparse
import pandas as pd

def predict_text(text):
    vec = joblib.load("models/vectorizer.pkl")
    model = joblib.load("models/model.pkl")
    X = vec.transform([text])
    prob = model.predict_proba(X)[0,1]
    label = "spam" if prob > 0.5 else "ham"
    print(f"Prediction: {label} (prob={prob:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    predict_text(args.text)
