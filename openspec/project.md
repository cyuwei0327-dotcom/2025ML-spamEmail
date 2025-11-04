# Project: 2025ML-spamEmail

## Purpose
以 TF-IDF + Logistic Regression 建立 SMS/Email 垃圾郵件二元分類器，提供 CLI 與 Streamlit 介面，且流程可複製、可視覺化、可部署。

## Scope
- 前處理（清洗、正規化、輸出乾淨 CSV）
- 訓練（向量化 + LR）
- 評估（Accuracy/Precision/Recall/F1、混淆矩陣、ROC/PR）
- 推論（單句與批次）
- Streamlit Demo（單句、批次、門檻、曲線、可下載）
- OpenSpec 規格與變更管理

## Non-Goals
- 深度學習/Transformers
- 雲端 API 與即時線上學習

## Tech Stack
- Python 3.10、pandas、scikit-learn、numpy、joblib、matplotlib、Streamlit
- Node.js LTS、OpenSpec CLI

## Layout
