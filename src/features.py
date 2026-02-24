import pandas as pd

def build_features(df):

    X = df[
        [
            "price",
            "freight_value",
            "delivery_days",
            "is_delayed"
        ]
    ]

    y = df["bad_review"]

    return X, y