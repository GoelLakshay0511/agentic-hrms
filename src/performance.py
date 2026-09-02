"""
Agentic HRMS — Performance Analytics Module
Handles performance classification, aggregation, and employee profile generation.
"""
import pandas as pd
import numpy as np


def classify_performance(score, scale="100"):
    """
    Classify a performance score into a category.
    scale: '100' for 0-100 range, '5' for 1-5 range, 'text' for text labels.
    """
    if isinstance(score, str):
        mapping = {
            "Exceeds": "Excellent",
            "Fully Meets": "Good",
            "Needs Improvement": "Needs Improvement",
            "PIP": "Needs Improvement",
        }
        return mapping.get(score, "Average")

    if scale == "5":
        if score >= 4.5:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 2.5:
            return "Average"
        else:
            return "Needs Improvement"
    else:  # 0-100 scale
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Average"
        else:
            return "Needs Improvement"


def get_performance_color(category):
    """Return color for performance category."""
    return {
        "Excellent": "#00C853",
        "Good": "#00D4FF",
        "Average": "#FFB300",
        "Needs Improvement": "#FF1744",
    }.get(category, "#8899AA")


def compute_performance_summary(df):
    """
    Compute summary statistics from Employee_Performance_Dataset.csv.
    Returns dict of aggregated metrics.
    """
    summary = {}

    if "Performance Score" in df.columns:
        summary["avg_performance"] = round(df["Performance Score"].mean(), 1)
        summary["median_performance"] = round(df["Performance Score"].median(), 1)

        # Classify each employee
        df = df.copy()
        df["Performance Category"] = df["Performance Score"].apply(
            lambda x: classify_performance(x, "100")
        )
        summary["category_counts"] = df["Performance Category"].value_counts().to_dict()

    if "KPI Score" in df.columns:
        summary["avg_kpi"] = round(df["KPI Score"].mean(), 1)

    if "Attendance (%)" in df.columns:
        summary["avg_attendance"] = round(df["Attendance (%)"].mean(), 1)

    if "Task Completion (%)" in df.columns:
        summary["avg_task_completion"] = round(df["Task Completion (%)"].mean(), 1)

    if "Training Hours" in df.columns:
        summary["avg_training_hours"] = round(df["Training Hours"].mean(), 1)

    if "Peer Rating" in df.columns:
        summary["avg_peer_rating"] = round(df["Peer Rating"].mean(), 2)

    if "Manager Feedback" in df.columns:
        summary["avg_manager_feedback"] = round(df["Manager Feedback"].mean(), 2)

    if "Promotion Eligibility" in df.columns:
        promo = df["Promotion Eligibility"].value_counts()
        summary["promo_eligible"] = promo.get("Yes", 0)
        summary["promo_not_eligible"] = promo.get("No", 0)

    return summary


def get_department_performance(df):
    """Get average performance metrics grouped by department."""
    dept_col = "Department" if "Department" in df.columns else None
    if dept_col is None:
        return pd.DataFrame()

    agg_cols = {}
    if "Performance Score" in df.columns:
        agg_cols["Performance Score"] = "mean"
    if "KPI Score" in df.columns:
        agg_cols["KPI Score"] = "mean"
    if "Attendance (%)" in df.columns:
        agg_cols["Attendance (%)"] = "mean"
    if "Task Completion (%)" in df.columns:
        agg_cols["Task Completion (%)"] = "mean"
    if "Training Hours" in df.columns:
        agg_cols["Training Hours"] = "mean"

    if not agg_cols:
        return pd.DataFrame()

    result = df.groupby(dept_col).agg(agg_cols).round(2).reset_index()
    return result


def get_employee_performance_profile(df, employee_id):
    """Get detailed performance profile for a single employee."""
    id_col = "Employee ID" if "Employee ID" in df.columns else "EmployeeID"
    row = df[df[id_col] == employee_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
