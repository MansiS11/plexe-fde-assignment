import shap
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/processed.csv").iloc[:100]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df.drop("bad_review", axis=1))

print("SHAP explanation generated")