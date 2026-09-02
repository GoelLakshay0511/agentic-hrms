"""
Agentic HRMS — Attrition Prediction ML Pipeline
Trains Logistic Regression + Random Forest, compares, and provides prediction utilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

from src.preprocessing import preprocess_attrition_data, prepare_single_prediction

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@st.cache_resource(show_spinner="Training attrition models...")
def train_attrition_models(df):
    """
    Train and compare Logistic Regression and Random Forest on attrition data.
    Returns the best model along with metrics and artifacts.
    """
    X_train, X_test, y_train, y_test, feature_names, encoders, scaler = preprocess_attrition_data(df)

    # ── Train Logistic Regression ───────────────────
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)

    # ── Train Random Forest ─────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate both ───────────────────────────────
    results = {}
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name]["fpr"] = fpr
        results[name]["tpr"] = tpr

    # ── Select best model (by F1 — better for imbalanced data) ────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]

    # ── Feature importance ──────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    # ── Save best model ─────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "attrition_model.joblib")
    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler": scaler,
        "best_name": best_name,
    }, model_path)

    return {
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler": scaler,
        "importance_df": importance_df,
    }


def predict_attrition_risk(model_artifacts, input_features):
    """
    Predict attrition probability for given input features.
    input_features: dict of raw feature values.
    Returns probability and risk level.
    """
    model = model_artifacts["best_model"]
    feature_names = model_artifacts["feature_names"]
    encoders = model_artifacts["encoders"]
    scaler = model_artifacts["scaler"]

    X = prepare_single_prediction(input_features, feature_names, encoders, scaler)
    proba = model.predict_proba(X)[0][1]

    if proba >= 0.7:
        risk = "HIGH"
    elif proba >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return proba, risk


def get_top_risk_factors(model_artifacts, n=10):
    """Return top N risk factors from the trained model."""
    return model_artifacts["importance_df"].head(n)


def get_feature_ranges(df):
    """
    Get min/max/unique values for each feature to power the what-if sliders.
    Returns dict: {feature_name: {type, min, max, values, default}}
    """
    # Drop non-feature columns
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Attrition"]
    cols = [c for c in df.columns if c not in drop_cols]

    ranges = {}
    for col in cols:
        if df[col].dtype == "object":
            vals = sorted(df[col].dropna().unique().tolist())
            ranges[col] = {
                "type": "categorical",
                "values": vals,
                "default": vals[0] if vals else "",
            }
        else:
            ranges[col] = {
                "type": "numerical",
                "min": int(df[col].min()),
                "max": int(df[col].max()),
                "default": int(df[col].median()),
            }
    return ranges
