# Capability: app

## ADDED Requirements

### Requirement: Explanations for single text
The app **MUST** explain why a single input is predicted as spam/ham by listing the most influential tokens.
#### Scenario: Show token contributions
WHEN the user enters a message and clicks **Predict**  
THEN the app shows the predicted probability and label  
AND lists the Top-K token contributions, where a positive sign pushes toward **spam** and a negative sign toward **ham**.

### Requirement: Batch CSV predictions with download
The app **SHALL** support batch CSV inference and **SHALL** allow users to download results enriched with probabilities, labels, and token explanations.
#### Scenario: Upload and predict
WHEN the user uploads a CSV and chooses a text column  
THEN the app computes `pred_prob` and `pred_label` for each row  
AND, when explanations are enabled, adds a `top_tokens` column with Top-K `token:weight` pairs  
AND provides a **Download CSV** button to export the results.

### Requirement: Model card panel
The app **MUST** surface basic model metadata so users can understand dataset, parameters, metrics, and training time.
#### Scenario: Show model card from file
WHEN `models/model_card.json` exists  
THEN the app displays dataset, parameters, metrics, classes, n_train/n_test, and trained_at in a readable JSON panel.

### Requirement: Adjustable threshold
The app **MUST** allow adjusting the decision threshold that converts probability to spam/ham labels.
#### Scenario: Change threshold
WHEN the user adjusts the threshold slider to a value **T**  
THEN all predictions (single and batch) label a row as **spam** iff `prob >= T`.
