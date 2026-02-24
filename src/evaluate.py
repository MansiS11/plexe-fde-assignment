import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from features import build_features
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed.csv")

X, y = build_features(df)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/model.pkl")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

errors = X_test.copy()
errors["true"] = y_test.values
errors["pred"] = preds

mistakes = errors[errors["true"] != errors["pred"]]

print(mistakes.head())