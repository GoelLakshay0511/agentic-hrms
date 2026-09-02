"""
Agentic HRMS — Data Loading & Global Filter Management
Cached loaders for all 7 CSV datasets + session-state-based global filtering.
"""
import streamlit as st
import pandas as pd
import os

# ─── Base Data Directory ──────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data(show_spinner=False)
def load_hr_data():
    """Load Cleaned_HR_Data_Analysis.csv — main employee dataset."""
    path = os.path.join(DATA_DIR, "Cleaned_HR_Data_Analysis.csv")
    df = pd.read_csv(path)
    # Clean whitespace in DepartmentType
    df["DepartmentType"] = df["DepartmentType"].str.strip()
    # Parse dates
    for col in ["StartDate", "DOB", "Survey Date", "Training Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="mixed")
    return df


@st.cache_data(show_spinner=False)
def load_attrition_data():
    """Load employee_attrition.csv — attrition prediction dataset."""
    path = os.path.join(DATA_DIR, "employee_attrition.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_performance_data():
    """Load Employee_Performance_Dataset.csv — performance analytics."""
    path = os.path.join(DATA_DIR, "Employee_Performance_Dataset.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_performance_pro_data():
    """Load employee_performance_pro.csv — extended employee profiles."""
    path = os.path.join(DATA_DIR, "employee_performance_pro.csv")
    df = pd.read_csv(path)
    # Parse joining date
    if "JoiningDate" in df.columns:
        df["JoiningDate"] = pd.to_datetime(df["JoiningDate"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_occupation_data():
    """Load occupation_data.csv — O*NET occupations."""
    path = os.path.join(DATA_DIR, "occupation_data.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_essential_skills():
    """Load essential_skills.csv — O*NET essential skills per occupation."""
    path = os.path.join(DATA_DIR, "essential_skills.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_software_skills():
    """Load software_skills.csv — O*NET software/technology per occupation."""
    path = os.path.join(DATA_DIR, "software_skills.csv")
    df = pd.read_csv(path)
    return df


# ─── Global Filter State ─────────────────────────────────────────────────────

def init_filter_state():
    """Initialize global filter defaults in session_state."""
    defaults = {
        "filter_department": "All",
        "filter_business_unit": "All",
        "filter_job_role": "All",
        "filter_status": "All",
        "selected_employee_id": None,
        "selected_employee_source": None,
        "selected_target_occupation": None,
        "live_mode": False,
        "last_refresh": None,
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_global_filters(df, location="sidebar"):
    """Render global filter controls. Returns filtered dataframe."""
    container = st.sidebar if location == "sidebar" else st

    # Determine available filter columns
    dept_col = None
    bu_col = None
    role_col = None
    status_col = None

    if "DepartmentType" in df.columns:
        dept_col = "DepartmentType"
    elif "Department" in df.columns:
        dept_col = "Department"

    if "BusinessUnit" in df.columns:
        bu_col = "BusinessUnit"

    if "Title" in df.columns and df["Title"].nunique() <= 50:
        role_col = "Title"
    elif "JobRole" in df.columns:
        role_col = "JobRole"
    elif "Job Role" in df.columns:
        role_col = "Job Role"

    if "EmployeeStatus" in df.columns:
        status_col = "EmployeeStatus"

    container.markdown("#### 🔍 Filters")

    # Department
    if dept_col:
        depts = ["All"] + sorted(df[dept_col].dropna().unique().tolist())
        idx = 0
        if st.session_state.get("filter_department", "All") in depts:
            idx = depts.index(st.session_state["filter_department"])
        sel = container.selectbox("Department", depts, index=idx, key="widget_filter_dept")
        st.session_state["filter_department"] = sel

    # Business Unit
    if bu_col:
        bus = ["All"] + sorted(df[bu_col].dropna().unique().tolist())
        idx = 0
        if st.session_state.get("filter_business_unit", "All") in bus:
            idx = bus.index(st.session_state["filter_business_unit"])
        sel = container.selectbox("Business Unit", bus, index=idx, key="widget_filter_bu")
        st.session_state["filter_business_unit"] = sel

    # Job Role
    if role_col:
        roles = ["All"] + sorted(df[role_col].dropna().unique().tolist())
        idx = 0
        if st.session_state.get("filter_job_role", "All") in roles:
            idx = roles.index(st.session_state["filter_job_role"])
        sel = container.selectbox("Job Role", roles, index=idx, key="widget_filter_role")
        st.session_state["filter_job_role"] = sel

    # Status
    if status_col:
        stats = ["All"] + sorted(df[status_col].dropna().unique().tolist())
        idx = 0
        if st.session_state.get("filter_status", "All") in stats:
            idx = stats.index(st.session_state["filter_status"])
        sel = container.selectbox("Status", stats, index=idx, key="widget_filter_status")
        st.session_state["filter_status"] = sel

    return apply_global_filters(df)


def apply_global_filters(df):
    """Apply current global filter state to a dataframe."""
    filtered = df.copy()

    dept_val = st.session_state.get("filter_department", "All")
    bu_val = st.session_state.get("filter_business_unit", "All")
    role_val = st.session_state.get("filter_job_role", "All")
    status_val = st.session_state.get("filter_status", "All")

    if dept_val != "All":
        for col in ["DepartmentType", "Department"]:
            if col in filtered.columns:
                filtered = filtered[filtered[col] == dept_val]
                break

    if bu_val != "All" and "BusinessUnit" in filtered.columns:
        filtered = filtered[filtered["BusinessUnit"] == bu_val]

    if role_val != "All":
        for col in ["Title", "JobRole", "Job Role"]:
            if col in filtered.columns:
                filtered = filtered[filtered[col] == role_val]
                break

    if status_val != "All" and "EmployeeStatus" in filtered.columns:
        filtered = filtered[filtered["EmployeeStatus"] == status_val]

    return filtered


def get_all_datasets():
    """Load all datasets and return as a dict. Cached at individual level."""
    return {
        "hr": load_hr_data(),
        "attrition": load_attrition_data(),
        "performance": load_performance_data(),
        "performance_pro": load_performance_pro_data(),
        "occupations": load_occupation_data(),
        "essential_skills": load_essential_skills(),
        "software_skills": load_software_skills(),
    }
