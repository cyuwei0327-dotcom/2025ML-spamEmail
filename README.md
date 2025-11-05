2025ML-SpamEmail (HW3: Email Spam USING OpenSpec)

This project implements an SMS / Email Spam Classifier using TF-IDF + Logistic Regression.
It includes both a CLI interface and an interactive Streamlit web app, with full OpenSpec documentation and change tracking.

Overview

The goal is to classify whether a message is Spam or Ham (Not Spam) using a machine learning pipeline and visualize predictions interactively.
This project follows reproducible steps — from data preprocessing to deployment — and adopts OpenSpec for structured documentation of features and requirements.

Tech Stack

Language: Python 3.10

Libraries: pandas, numpy, scikit-learn, joblib, streamlit, matplotlib

Tools: OpenSpec CLI (for structured specs)

Deployment: Streamlit Cloud

Version Control: GitHub

Folder Structure
2025ML-spamEmail/
├─ app/                     # Streamlit frontend
│   └─ streamlit_app.py
├─ scripts/                 # CLI for preprocessing, training, predicting
│   ├─ preprocess_emails.py
│   ├─ train_spam_classifier.py
│   └─ predict_spam.py
├─ datasets/
│   ├─ sms_spam_no_header.csv
│   └─ clean.csv
├─ models/
│   ├─ model.pkl
│   └─ vectorizer.pkl
├─ openspec/                # OpenSpec project & changes
│   ├─ project.md
│   ├─ specs/app/spec.md
│   └─ changes/add-explanations-and-downloads/
│       ├─ proposal.md
│       ├─ tasks.md
│       └─ specs/app/spec.md
├─ requirements.txt
└─ README.md

How to Run Locally
1. Clone this repo
git clone https://github.com/cyuwei0327-dotcom/2025ML-spamEmail.git
cd 2025ML-spamEmail

2. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Preprocess & Train
python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/clean.csv
python scripts/train_spam_classifier.py --input datasets/clean.csv

5. Predict (CLI)
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash" --explain --topk 5
python scripts/predict_spam.py --input datasets/clean.csv --output predictions.csv --explain --topk 5

6. Run the Streamlit App
streamlit run app/streamlit_app.py


Then open your browser → http://localhost:8501

Demo Site

Hosted on Streamlit Cloud
 https://2025spamemail.streamlit.app/

OpenSpec Integration

This project uses OpenSpec for structured documentation of software requirements.

Commands
# Validate current change
openspec validate add-explanations-and-downloads --strict

# Archive change after validation
openspec archive add-explanations-and-downloads --yes

Specification Path
openspec/
 ├─ project.md
 ├─ specs/app/spec.md
 └─ changes/add-explanations-and-downloads/


Latest Archived Change: 2025-11-04-add-explanations-and-downloads
Features added:

Token contribution explanations (--explain, --topk)

Batch CSV upload and downloadable predictions

Model card visualization

Adjustable probability threshold

Example Output

CLI prediction example:

Prediction: spam  (prob=0.923, threshold=0.50)
Top tokens: free:0.431, win:0.225, cash:0.172, entry:0.167, comp:0.143


Streamlit Interface:

Enter text to predict spam/ham

Upload CSV for batch analysis

Adjust threshold slider

Download predicted results as CSV

References

Dataset: PacktPublishing / Hands-On Artificial Intelligence for Cybersecurity

Tutorial Video: YouTube – Streamlit Spam Email Classifier Demo