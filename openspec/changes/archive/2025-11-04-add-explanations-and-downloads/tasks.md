## 1) CLI explanations
- [ ] 新增 `--explain --topk` 至 scripts/predict_spam.py
- [ ] 計算 TF-IDF × coef_ 簽名貢獻，單句/批次皆可輸出
- [ ] 批次模式加入 `top_tokens` 欄位

## 2) Streamlit 批次與下載
- [x] CSV 上傳、欄位選擇
- [x] 顯示 `pred_prob`、`pred_label`、`top_tokens`
- [x] `st.download_button()` 匯出結果

## 3) Model Card
- [x] 訓練時寫 models/model_card.json
- [x] UI 顯示 model card

## 4) Docs & Validation
- [ ] 更新 README
- [ ] `openspec validate add-explanations-and-downloads --strict`
