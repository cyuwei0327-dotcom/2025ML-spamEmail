import pandas as pd
import argparse
import re

CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

def try_read_csv(path):
    last_err = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            # sep=None + engine="python" 可自動偵測逗號/分號/Tab
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
            print(f"[info] loaded with encoding={enc}, shape={df.shape}")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def normalize_to_two_cols(df):
    """
    兼容各種格式：
    1) 沒有表頭：兩欄（label, text）
    2) Kaggle sms spam（常見 header v1, v2 或還有多餘欄位）
    3) 如果超過兩欄 → 只取前兩欄
    """
    # 嘗試常見欄名
    candidates = [
        ("v1", "v2"),
        ("label", "text"),
        ("Category", "Message"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            sub = df[[a, b]].copy()
            sub.columns = ["label", "text"]
            return sub

    # 沒有對得上的欄名 → 用位置取前兩欄
    if df.shape[1] >= 2:
        sub = df.iloc[:, :2].copy()
        sub.columns = ["label", "text"]
        return sub

    # 最後手段：當成沒有表頭重讀一次
    df2 = pd.read_csv(args.input, header=None, names=["label", "text"], encoding="latin-1", sep=None, engine="python", on_bad_lines="skip")
    return df2

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main(args):
    df_raw = try_read_csv(args.input)
    df = normalize_to_two_cols(df_raw)

    # 去掉缺失與空字串
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip()
    df["text"]  = df["text"].astype(str).str.strip()

    # 清洗文字
    df["text_clean"] = df["text"].apply(clean_text)

    # 輸出
    df[["label", "text", "text_clean"]].to_csv(args.output, index=False, encoding="utf-8")
    print(f"Cleaned data saved to {args.output} (rows={len(df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
