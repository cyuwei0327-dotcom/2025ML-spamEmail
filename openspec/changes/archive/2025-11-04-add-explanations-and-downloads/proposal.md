## Change ID
add-explanations-and-downloads

## Why
需要讓使用者看懂「為什麼是 spam」，並支援批次上傳與下載結果用於報告。

## What
- Streamlit：批次上傳 CSV、顯示 `pred_prob`/`pred_label` 與 **Top-K token 貢獻**、提供 **下載 CSV**
- CLI：`predict_spam.py` 支援 `--explain --topk`，輸出每列 top tokens 與權重
- 訓練：輸出 `models/model_card.json`，UI 顯示 Model Card

## Impact
app/streamlit_app.py、scripts/predict_spam.py、scripts/train_spam_classifier.py（新增 model_card）

## Risks
scikit-learn 版本需支援 `get_feature_names_out()`；建議 >= 1.0。
