"""
Agentic HRMS — Data Preprocessing for ML Models
Handles cleaning, encoding, scaling for the attrition prediction pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_attrition_data(df):
    """
    Preprocess employee_attrition.csv for ML training.
    Returns X_train, X_test, y_train, y_test, feature_names, encoders, scaler
    """
    df = df.copy()

    # ── Target encoding ─────────────────────────────
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # ── Drop constant / ID columns ──────────────────
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Handle missing values ───────────────────────
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ── Encode categoricals ─────────────────────────
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Attrition" in cat_cols:
        cat_cols.remove("Attrition")

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── Separate features and target ────────────────
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    feature_names = X.columns.tolist()

    # ── Scale numerical features ────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # ── Train/test split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, encoders, scaler


def prepare_single_prediction(input_dict, feature_names, encoders, scaler):
    """
    Prepare a single employee's features for prediction.
    input_dict: dict of raw feature values (before encoding).
    Returns scaled feature array ready for model.predict_proba().
    """
    row = {}
    for feat in feature_names:
        if feat in input_dict:
            val = input_dict[feat]
            if feat in encoders:
                le = encoders[feat]
                if val in le.classes_:
                    val = le.transform([val])[0]
                else:
                    # Unseen category: use most frequent
                    val = 0
            row[feat] = val
        else:
            row[feat] = 0  # Default for missing features

    df_row = pd.DataFrame([row], columns=feature_names)
    df_scaled = pd.DataFrame(scaler.transform(df_row), columns=feature_names)
    return df_scaled
