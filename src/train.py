import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from features import build_features

# Load data
df = pd.read_csv("data/processed.csv")

# Features
X, y = build_features(df)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("✅ Model trained and saved")