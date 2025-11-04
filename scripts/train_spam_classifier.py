import os
import argparse
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import numpy as np

def main(args):
    df = pd.read_csv(args.input)

    # 保障欄位存在
    assert {"label", "text_clean"}.issubset(df.columns), "clean.csv 需要包含 label, text_clean 欄位"

    # 清理：轉字串、去空白、移除空值與空字串
    df["label"] = df["label"].astype(str).str.strip()
    df["text_clean"] = df["text_clean"].astype(str).str.strip()
    df = df.dropna(subset=["label", "text_clean"])
    df = df[df["text_clean"].str.len() > 0]

    # （可選）標籤正規化
    df["label"] = df["label"].str.lower().replace({"ham": "ham", "spam": "spam"})
    # 若你的資料是 0/1，可改成：
    # df["label"] = df["label"].replace({0: "ham", 1: "spam"}).astype(str)

    X = df["text_clean"].values
    y = df["label"].values

    # 切分資料（用 stratify 維持類別比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, n_jobs=None)
    clf.fit(X_train_vec, y_train)

    # 評估
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="spam")

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}  Precision(spam): {prec:.4f}  Recall(spam): {rec:.4f}  F1(spam): {f1:.4f}")

    # 儲存模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(clf, "models/model.pkl")

    # 簡易 model card（之後可在 Streamlit 顯示）
    card = {
        "dataset": args.input,
        "seed": 42,
        "vectorizer": {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True},
        "model": {"type": "LogisticRegression", "C": 2.0, "class_weight": "balanced", "max_iter": 1000},
        "metrics": {"accuracy": float(acc), "precision_spam": float(prec), "recall_spam": float(rec), "f1_spam": float(f1)},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "classes": sorted(list(np.unique(y)))
    }
    with open("models/model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    print("Model, vectorizer, and model_card.json saved to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    main(args)
