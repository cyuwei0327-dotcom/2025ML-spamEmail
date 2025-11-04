import io
import json
import os
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

@st.cache_resource
def load_artifacts(models_dir: str = "models"):
    vec_path = os.path.join(models_dir, "vectorizer.pkl")
    mdl_path = os.path.join(models_dir, "model.pkl")
    card_path = os.path.join(models_dir, "model_card.json")

    vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None
    model = joblib.load(mdl_path) if os.path.exists(mdl_path) else None
    model_card = None
    if os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as f:
            model_card = json.load(f)
    return vectorizer, model, model_card


def ensure_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return s


def label_from_prob(p: float, threshold: float) -> str:
    return "spam" if p >= threshold else "ham"


def explain_top_tokens(texts: List[str], vectorizer, clf, topk: int = 5) -> List[List[Tuple[str, float]]]:
    X = vectorizer.transform(texts)
    coef = clf.coef_[0]
    vocab = np.array(vectorizer.get_feature_names_out())
    results = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        nz_idx = row.indices
        nz_val = row.data
        contrib = coef[nz_idx] * nz_val  # signed contribution
        if contrib.size == 0:
            results.append([])
            continue
        order = np.argsort(-np.abs(contrib))[:topk]
        top = [(vocab[nz_idx[j]], float(contrib[j])) for j in order]
        results.append(top)
    return results


def plot_confusion_matrix(y_true, y_pred, labels=("ham", "spam")):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true, y_score):
    # Convert labels to binary {spam:1, ham:0}
    y_bin = np.array([1 if y == "spam" else 0 for y in y_true])

    # ROC
    fpr, tpr, _ = roc_curve(y_bin, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    fig_roc.tight_layout()

    # PR
    precision, recall, _ = precision_recall_curve(y_bin, y_score)
    pr_auc = auc(recall, precision)
    fig_pr, ax2 = plt.subplots()
    ax2.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    fig_pr.tight_layout()

    return fig_roc, fig_pr, roc_auc, pr_auc


def class_report(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="spam")
    return {"accuracy": acc, "precision_spam": prec, "recall_spam": rec, "f1_spam": f1}


def autodetect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["text_clean", "text", "message", "content", "sms", "email"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: pick the last object column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[-1] if obj_cols else None


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")
st.title("📧 Spam Email Classifier — Rich Demo")

with st.sidebar:
    st.markdown("### Artifacts")
    vec, model, model_card = load_artifacts()
    if model is None or vec is None:
        st.error("Models not found. 請先在本機執行訓練腳本：\n\n`python scripts/train_spam_classifier.py --input datasets/clean.csv`")
    else:
        st.success("Model & Vectorizer loaded.")

    threshold = st.slider("Decision Threshold (spam if prob ≥ threshold)", 0.1, 0.9, 0.50, 0.01)
    topk = st.number_input("Top-K token explanations", min_value=0, max_value=20, value=5, step=1)

tabs = st.tabs(["🔮 Quick Predict", "🗂️ Batch Predict", "📊 Metrics & Curves", "🧭 Dataset Explorer", "📜 Model Card"])

# -----------------------------
# Tab 1: Quick Predict
# -----------------------------
with tabs[0]:
    st.subheader("Quick Predict")
    text = st.text_area("Enter an email/SMS message:", height=120, placeholder="e.g., Free entry in 2 a wkly comp to win cash")
    go = st.button("Predict", type="primary", use_container_width=False)
    if go:
        if model is None or vec is None:
            st.warning("Model not loaded.")
        else:
            X = vec.transform([text])
            prob = float(model.predict_proba(X)[0, 1])
            label = label_from_prob(prob, threshold)
            st.metric("Prediction", "SPAM 🚨" if label == "spam" else "HAM ✅", help=f"prob={prob:.3f}, threshold={threshold:.2f}")

            if topk > 0:
                toks = explain_top_tokens([text], vec, model, topk=topk)[0]
                if toks:
                    st.markdown("**Top token contributions** (正值→推向 spam / 負值→推向 ham)")
                    st.dataframe(pd.DataFrame(toks, columns=["token", "contribution"]))
                else:
                    st.info("No token contributions (empty).")


# -----------------------------
# Tab 2: Batch Predict
# -----------------------------
with tabs[1]:
    st.subheader("Batch Predict with CSV")
    up = st.file_uploader("Upload CSV (choose a text column)", type=["csv"])
    text_col_name = None
    df_pred = None
    if up is not None:
        df_up = pd.read_csv(up)
        st.write("Preview:", df_up.head())
        text_col_name = autodetect_text_column(df_up)
        text_col_name = st.selectbox("Text column", options=list(df_up.columns), index=(list(df_up.columns).index(text_col_name) if text_col_name in df_up.columns else 0))
        if st.button("Run Batch Predict", type="primary"):
            if model is None or vec is None:
                st.warning("Model not loaded.")
            else:
                texts = ensure_text_series(df_up[text_col_name]).tolist()
                X = vec.transform(texts)
                probs = model.predict_proba(X)[:, 1]
                labels = [label_from_prob(p, threshold) for p in probs]
                df_pred = df_up.copy()
                df_pred["pred_prob"] = probs
                df_pred["pred_label"] = labels
                if topk > 0:
                    exps = explain_top_tokens(texts, vec, model, topk=topk)
                    df_pred["top_tokens"] = [", ".join([f"{t}:{w:.3f}" for t, w in xs]) for xs in exps]
                st.success(f"Done. {len(df_pred)} rows.")
                st.dataframe(df_pred.head(30))

                buf = io.StringIO()
                df_pred.to_csv(buf, index=False)
                st.download_button("Download predictions CSV", data=buf.getvalue().encode("utf-8"), file_name="predictions_with_explanations.csv", mime="text/csv")


# -----------------------------
# Tab 3: Metrics & Curves
# -----------------------------
with tabs[2]:
    st.subheader("Metrics & Curves")
    colA, colB = st.columns(2)
    source = st.radio("Evaluation data source", ["Use datasets/clean.csv", "Upload CSV"], horizontal=True)

    eval_df = None
    if source == "Use datasets/clean.csv":
        path = "datasets/clean.csv"
        if os.path.exists(path):
            eval_df = pd.read_csv(path)
        else:
            st.info("找不到 datasets/clean.csv，請改用上傳模式或先跑前處理腳本。")
    else:
        uploaded = st.file_uploader("Upload labeled CSV (needs label & text_clean columns)", type=["csv"], key="eval_up")
        if uploaded is not None:
            eval_df = pd.read_csv(uploaded)

    if eval_df is not None:
        # Try to find label/text columns
        label_col = "label" if "label" in eval_df.columns else st.selectbox("Label column", options=list(eval_df.columns))
        text_col = "text_clean" if "text_clean" in eval_df.columns else autodetect_text_column(eval_df)
        text_col = st.selectbox("Text column", options=list(eval_df.columns), index=list(eval_df.columns).index(text_col) if text_col in eval_df.columns else 0)

        if st.button("Evaluate", type="primary"):
            if model is None or vec is None:
                st.warning("Model not loaded.")
            else:
                dfE = eval_df.dropna(subset=[label_col, text_col]).copy()
                dfE[label_col] = dfE[label_col].astype(str).str.strip().str.lower().replace({"0": "ham", "1": "spam"})
                texts = ensure_text_series(dfE[text_col]).tolist()

                X = vec.transform(texts)
                probs = model.predict_proba(X)[:, 1]
                preds = [label_from_prob(p, threshold) for p in probs]

                metrics = class_report(dfE[label_col], preds)
                colA.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                colA.metric("Precision(spam)", f"{metrics['precision_spam']:.3f}")
                colB.metric("Recall(spam)", f"{metrics['recall_spam']:.3f}")
                colB.metric("F1(spam)", f"{metrics['f1_spam']:.3f}")

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    fig_cm = plot_confusion_matrix(dfE[label_col], preds, labels=("ham", "spam"))
                    st.pyplot(fig_cm, clear_figure=True)
                with c2:
                    fig_roc, fig_pr, roc_auc, pr_auc = plot_roc_pr(dfE[label_col], probs)
                    st.pyplot(fig_roc, clear_figure=True)
                    st.pyplot(fig_pr, clear_figure=True)


# -----------------------------
# Tab 4: Dataset Explorer
# -----------------------------
with tabs[3]:
    st.subheader("Dataset Explorer")
    chosen = st.radio("Data source", ["Use datasets/clean.csv", "Upload CSV"], horizontal=True, key="explore_src")
    dfX = None
    if chosen == "Use datasets/clean.csv":
        if os.path.exists("datasets/clean.csv"):
            dfX = pd.read_csv("datasets/clean.csv")
        else:
            st.info("沒有 datasets/clean.csv，可改用上傳。")
    else:
        upX = st.file_uploader("Upload CSV", type=["csv"], key="explore_up")
        if upX is not None:
            dfX = pd.read_csv(upX)

    if dfX is not None:
        st.write("Preview:", dfX.head())
        # label dist
        if "label" in dfX.columns:
            st.markdown("**Label distribution**")
            st.bar_chart(dfX["label"].value_counts())
        # length dist
        txt_col = "text_clean" if "text_clean" in dfX.columns else autodetect_text_column(dfX)
        if txt_col:
            st.markdown(f"**Length distribution** ({txt_col})")
            lens = ensure_text_series(dfX[txt_col]).str.split().apply(len)
            st.line_chart(lens.value_counts().sort_index())


# -----------------------------
# Tab 5: Model Card
# -----------------------------
with tabs[4]:
    st.subheader("Model Card")
    if model_card is None:
        st.info("No model_card.json found in models/. 訓練腳本會自動生成。")
    else:
        st.json(model_card)
