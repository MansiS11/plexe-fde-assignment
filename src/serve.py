from fastapi import FastAPI
import joblib
import pandas as pd
import shap

app = FastAPI(title="Marketplace Risk Predictor")

model = joblib.load("models/model.pkl")
explainer = shap.TreeExplainer(model)

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]

    shap_values = explainer.shap_values(df)[0]

    feature_imp = sorted(
        zip(df.columns, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return {
        "bad_review_probability": float(prob),
        "top_features": feature_imp
    }