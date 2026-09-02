"""
Agentic HRMS — Workforce Intelligence Platform
Main Streamlit Application with 9 interactive pages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random

# ─── Page Config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Agentic HRMS — Workforce Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports ─────────────────────────────────────────────────────────────────
from src.utils import (
    inject_custom_css, render_kpi_card, render_status_pill,
    render_section_header, render_footer, apply_chart_style,
    create_gauge_chart, COLORS, get_risk_color, render_sidebar_status,
)
from src.data_loader import (
    load_hr_data, load_attrition_data, load_performance_data,
    load_performance_pro_data, load_occupation_data,
    load_essential_skills, load_software_skills,
    init_filter_state, render_global_filters, apply_global_filters,
)
from src.attrition_model import (
    train_attrition_models, predict_attrition_risk,
    get_top_risk_factors, get_feature_ranges,
)
from src.performance import (
    classify_performance, get_performance_color,
    compute_performance_summary, get_department_performance,
)
from src.skill_gap import (
    get_occupation_titles, find_occupation_by_title,
    get_combined_skills_for_occupation, infer_employee_skills,
    semantic_match_skills, calculate_readiness_score,
    get_essential_skills_for_occupation, get_software_skills_for_occupation,
)
from src.recommender import (
    recommend_courses_for_gaps, generate_learning_path,
    get_certification_suggestions, get_course_catalog,
)
from src.rag import build_rag_index, search_policies, format_rag_response
from src.agent import (
    detect_intent, execute_tools, generate_response,
    stream_response, TOOLS,
)

# ─── Initialize ──────────────────────────────────────────────────────────────
inject_custom_css()
init_filter_state()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 5px 0;">
        <span style="font-size:1.8rem;">🧠</span><br>
        <span style="color:#00D4FF; font-size:1rem; font-weight:700; letter-spacing:2px;">WORKFORCE</span><br>
        <span style="color:#8899AA; font-size:0.75rem; font-weight:500; letter-spacing:3px;">INTELLIGENCE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "◉ Dashboard",
            "◉ Employee Intelligence",
            "◉ Attrition Prediction",
            "◉ Performance",
            "◉ Skill Gap",
            "◉ Recommendations",
            "◉ Workforce",
            "◉ AI Assistant",
            "◉ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Global filters (for pages that need them)
    hr_df = load_hr_data()
    if page in ["◉ Dashboard", "◉ Employee Intelligence", "◉ Performance"]:
        filtered_hr = render_global_filters(hr_df)
    else:
        filtered_hr = hr_df

    st.markdown("---")

    # Live Mode toggle
    live_mode = st.toggle("🟢 Live Mode", value=st.session_state.get("live_mode", False), key="live_mode_toggle")
    st.session_state["live_mode"] = live_mode
    if live_mode:
        st.caption("⚡ Simulated live data for demo purposes")
    last_refresh = st.session_state.get("last_refresh") or datetime.now()
    st.caption(f"Last updated: {last_refresh.strftime('%H:%M:%S')}")

    # Status indicators
    render_sidebar_status(
        model_ready=True,
        data_ready=True,
        rag_ready=True,
        live_mode=live_mode,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "◉ Dashboard":
    st.markdown("""
    <div style="text-align:center; margin-bottom:8px;">
        <h1 style="margin-bottom:2px; font-size:2rem;">🧠 Workforce Intelligence Platform</h1>
        <p style="color:#6B7D8E; font-size:0.95rem;">AI-powered workforce analytics, skill intelligence and employee development</p>
    </div>
    """, unsafe_allow_html=True)

    # Live Mode refresh
    if live_mode:
        col_r1, col_r2 = st.columns([6, 1])
        with col_r2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state["last_refresh"] = datetime.now()
                st.rerun()
        with col_r1:
            st.info("📡 **Live Mode Active** — Simulated live data for demo purposes. Metrics include a small random perturbation.", icon="📡")

    # Compute KPIs from filtered data
    df = filtered_hr
    perf_df = load_performance_data()
    perf_pro_df = load_performance_pro_data()
    attr_df = load_attrition_data()

    total_emp = len(df)
    active_emp = len(df[df["EmployeeStatus"] == "Active"]) if "EmployeeStatus" in df.columns else total_emp

    avg_engagement = round(df["Engagement Score"].mean(), 2) if "Engagement Score" in df.columns else 0
    avg_satisfaction = round(df["Satisfaction Score"].mean(), 2) if "Satisfaction Score" in df.columns else 0
    avg_rating = round(df["Current Employee Rating"].mean(), 2) if "Current Employee Rating" in df.columns else 0

    # Attrition rate
    high_risk = perf_pro_df[perf_pro_df["AttritionRisk"] == "Yes"].shape[0]
    attr_rate = round(attr_df["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100, 1)

    # Training
    training_completed = 0
    if "Training Outcome" in df.columns:
        training_counts = df["Training Outcome"].value_counts()
        training_completed = round(training_counts.get("Completed", 0) / max(len(df), 1) * 100, 1)

    # Add live perturbation
    jitter = 0
    if live_mode:
        jitter = random.uniform(-0.05, 0.05)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("👥", f"{total_emp:,}", "Total Employees",
                       delta=f"{active_emp:,} active", delta_type="positive",
                       accent_color=COLORS["accent_cyan"])
    with col2:
        eng_val = round(avg_engagement * (1 + jitter), 2)
        render_kpi_card("💡", f"{eng_val}/5", "Avg Engagement",
                       delta="from surveys", delta_type="neutral",
                       accent_color=COLORS["accent_teal"])
    with col3:
        risk_val = int(high_risk * (1 + jitter * 0.5))
        render_kpi_card("⚠️", f"{risk_val}", "High Attrition Risk",
                       delta=f"{attr_rate}% overall rate", delta_type="negative",
                       accent_color=COLORS["danger"])
    with col4:
        render_kpi_card("📊", f"{avg_rating}/5", "Avg Performance Rating",
                       delta=f"Satisfaction: {avg_satisfaction}/5", delta_type="positive",
                       accent_color=COLORS["positive"])

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6, col7 = st.columns(3)
    with col5:
        render_kpi_card("🎓", f"{training_completed}%", "Training Completion",
                       accent_color="#7C4DFF")
    with col6:
        render_kpi_card("🏢", f"{df['DepartmentType'].nunique() if 'DepartmentType' in df.columns else '-'}",
                       "Departments", accent_color=COLORS["accent_cyan"])
    with col7:
        sat_val = round(avg_satisfaction * (1 + jitter), 2)
        render_kpi_card("😊", f"{sat_val}/5", "Avg Satisfaction",
                       accent_color=COLORS["warning"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if "DepartmentType" in df.columns:
            dept_counts = df["DepartmentType"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Count"]
            fig = px.bar(dept_counts, x="Department", y="Count",
                        title="Employee Distribution by Department",
                        color="Count",
                        color_continuous_scale=["#1B2838", "#00D4FF"])
            apply_chart_style(fig)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if "Performance Score" in df.columns:
            perf_counts = df["Performance Score"].value_counts().reset_index()
            perf_counts.columns = ["Performance", "Count"]
            colors = [get_performance_color(p) for p in perf_counts["Performance"]]
            fig = px.pie(perf_counts, values="Count", names="Performance",
                        title="Performance Score Distribution",
                        color_discrete_sequence=COLORS["chart_colors"])
            apply_chart_style(fig)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Training Outcome
        if "Training Outcome" in df.columns:
            tr_counts = df["Training Outcome"].value_counts().reset_index()
            tr_counts.columns = ["Outcome", "Count"]
            fig = px.bar(tr_counts, x="Outcome", y="Count",
                        title="Training Outcome Distribution",
                        color="Outcome",
                        color_discrete_sequence=COLORS["chart_colors"])
            apply_chart_style(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with chart_col4:
        # Engagement vs Satisfaction scatter
        if "Engagement Score" in df.columns and "Satisfaction Score" in df.columns:
            sample = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
            fig = px.scatter(sample, x="Engagement Score", y="Satisfaction Score",
                            color="Performance Score" if "Performance Score" in df.columns else None,
                            title="Engagement vs Satisfaction",
                            color_discrete_sequence=COLORS["chart_colors"],
                            opacity=0.6)
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Charts Row 3
    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        # Attrition distribution
        attr_counts = attr_df["Attrition"].value_counts().reset_index()
        attr_counts.columns = ["Attrition", "Count"]
        fig = px.pie(attr_counts, values="Count", names="Attrition",
                    title="Attrition Distribution",
                    color="Attrition",
                    color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["positive"]})
        apply_chart_style(fig)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with chart_col6:
        # Department-level performance from performance dataset
        dept_perf = get_department_performance(perf_df)
        if not dept_perf.empty and "Performance Score" in dept_perf.columns:
            fig = px.bar(dept_perf, x="Department", y="Performance Score",
                        title="Avg Performance Score by Department",
                        color="Performance Score",
                        color_continuous_scale=["#FF1744", "#FFB300", "#00C853"])
            apply_chart_style(fig)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Employee Intelligence":
    render_section_header("Employee Intelligence", "Comprehensive employee profile & analytics", "🔍")

    # Load datasets
    perf_pro = load_performance_pro_data()
    perf_df = load_performance_data()
    df = filtered_hr

    # Employee selection
    # Build a combined employee list
    employees = {}
    for _, row in perf_pro.iterrows():
        eid = row["EmployeeID"]
        name = row.get("Name", f"Employee {eid}")
        employees[eid] = f"{eid} — {name} ({row.get('JobRole', 'N/A')})"

    emp_ids = sorted(employees.keys())
    emp_labels = [employees[eid] for eid in emp_ids]

    # Default selection from session state
    default_idx = 0
    if st.session_state.get("selected_employee_id") in emp_ids:
        default_idx = emp_ids.index(st.session_state["selected_employee_id"])

    selected_label = st.selectbox(
        "🔍 Search Employee",
        emp_labels,
        index=default_idx,
        key="emp_intel_select",
    )
    selected_id = emp_ids[emp_labels.index(selected_label)]
    st.session_state["selected_employee_id"] = selected_id
    st.session_state["selected_employee_source"] = "performance_pro"

    # Get employee data
    emp = perf_pro[perf_pro["EmployeeID"] == selected_id].iloc[0]

    # Profile Header
    name = emp.get("Name", "Unknown")
    initials = "".join([n[0] for n in str(name).split()[:2]]).upper()

    st.markdown(f"""
    <div class="profile-header">
        <div class="profile-avatar">{initials}</div>
        <h2 style="margin:0; color:#FFFFFF;">{name}</h2>
        <p style="color:#00D4FF; margin:4px 0;">{emp.get('JobRole', 'N/A')} — {emp.get('Department', 'N/A')}</p>
        <p style="color:#6B7D8E; font-size:0.85rem;">ID: {selected_id} &nbsp;|&nbsp; {emp.get('Country', 'N/A')} &nbsp;|&nbsp; Age: {emp.get('Age', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        perf_rating = emp.get("PerformanceRating", 0)
        st.metric("Performance Rating", f"{perf_rating}/5",
                  delta=classify_performance(perf_rating, "5"))
    with col2:
        wlb = emp.get("WorkLifeBalanceScore", 0)
        st.metric("Work-Life Balance", f"{round(wlb, 1)}/5")
    with col3:
        cust_sat = emp.get("CustomerSatisfaction", None)
        st.metric("Customer Satisfaction", f"{cust_sat}/10" if pd.notna(cust_sat) else "N/A")
    with col4:
        risk = emp.get("AttritionRisk", "Unknown")
        risk_color = "#00C853" if risk == "No" else "#FF1744"
        st.metric("Attrition Risk", risk)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Info Cards
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("""<div class="info-card"><h4>📋 Employment Details</h4></div>""", unsafe_allow_html=True)
        detail_data = {
            "Education Level": emp.get("EducationLevel", "N/A"),
            "Years at Company": emp.get("YearsAtCompany", "N/A"),
            "Monthly Salary": f"${emp.get('MonthlySalary', 0):,}",
            "Overtime (hrs/month)": emp.get("OvertimeHoursPerMonth", "N/A"),
            "Projects Handled": emp.get("ProjectsHandled", "N/A"),
            "Training Hours": emp.get("TrainingHours", "N/A"),
            "Last Promotion": emp.get("LastPromotionYear", "N/A"),
            "Leaves Taken": emp.get("LeavesTaken", "N/A"),
        }
        for k, v in detail_data.items():
            st.markdown(f"**{k}:** {v}")

    with info_col2:
        st.markdown("""<div class="info-card"><h4>📊 Performance Indicators</h4></div>""", unsafe_allow_html=True)

        # Performance gauge
        perf_pct = min(perf_rating / 5 * 100, 100)
        fig = create_gauge_chart(perf_pct, "Performance Rating")
        st.plotly_chart(fig, use_container_width=True)

    # Check if employee exists in performance dataset too
    perf_match = perf_df[perf_df["Employee ID"] == selected_id]
    if not perf_match.empty:
        st.markdown("---")
        render_section_header("Performance Details", "From Employee Performance Dataset", "⭐")
        pr = perf_match.iloc[0]

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Performance Score", f"{pr.get('Performance Score', 'N/A')}/100")
        with pcol2:
            st.metric("KPI Score", f"{round(pr.get('KPI Score', 0), 1)}")
        with pcol3:
            st.metric("Attendance", f"{round(pr.get('Attendance (%)', 0), 1)}%")
        with pcol4:
            st.metric("Task Completion", f"{round(pr.get('Task Completion (%)', 0), 1)}%")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ATTRITION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Attrition Prediction":
    render_section_header("Attrition Prediction", "ML-powered employee attrition risk analysis", "⚠️")

    attr_df = load_attrition_data()

    # Train models (cached)
    with st.spinner("Training attrition models..."):
        model_artifacts = train_attrition_models(attr_df)

    results = model_artifacts["results"]
    best_name = model_artifacts["best_name"]

    # Model Comparison
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 What-If Simulator", "📈 Feature Analysis"])

    with tab1:
        st.markdown(f"**Best Model: {best_name}** (selected by F1 score)")

        # Metrics comparison table
        metrics_data = []
        for name, r in results.items():
            metrics_data.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']:.3f}",
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1 Score": f"{r['f1']:.3f}",
                "ROC-AUC": f"{r['roc_auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        # Charts
        cm_col, roc_col = st.columns(2)

        with cm_col:
            # Confusion Matrix
            cm = results[best_name]["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=["Stayed", "Left"], y=["Stayed", "Left"],
                           title=f"Confusion Matrix — {best_name}",
                           color_continuous_scale=["#1B2838", "#00D4FF"])
            apply_chart_style(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with roc_col:
            # ROC Curve
            fig = go.Figure()
            for name, r in results.items():
                fig.add_trace(go.Scatter(
                    x=r["fpr"], y=r["tpr"],
                    name=f"{name} (AUC={r['roc_auc']:.3f})",
                    mode="lines",
                ))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                     mode="lines", line=dict(dash="dash", color="#3A4A5A")))
            fig.update_layout(title="ROC Curve Comparison",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate")
            apply_chart_style(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.warning("⚠️ This is a predictive decision-support tool and should not be used as the sole basis for employment decisions.")

    with tab2:
        st.markdown("### 🔮 What-If Attrition Simulator")
        st.caption("Adjust employee features below and see the predicted attrition risk update in real-time.")

        feature_ranges = get_feature_ranges(attr_df)
        top_features = model_artifacts["importance_df"].head(12)["Feature"].tolist()

        # Build input controls for top features
        input_features = {}
        control_cols = st.columns(3)

        for i, feat in enumerate(top_features):
            col = control_cols[i % 3]
            if feat in feature_ranges:
                fr = feature_ranges[feat]
                with col:
                    if fr["type"] == "categorical":
                        val = st.selectbox(feat, fr["values"],
                                          index=0, key=f"whatif_{feat}")
                    else:
                        val = st.slider(feat,
                                       min_value=fr["min"],
                                       max_value=fr["max"],
                                       value=fr["default"],
                                       key=f"whatif_{feat}")
                    input_features[feat] = val

        # Fill remaining features with defaults
        for feat, fr in feature_ranges.items():
            if feat not in input_features:
                input_features[feat] = fr.get("default", fr.get("values", [""])[0] if fr["type"] == "categorical" else 0)

        # Predict
        proba, risk_level = predict_attrition_risk(model_artifacts, input_features)

        st.markdown("---")
        pred_col1, pred_col2, pred_col3 = st.columns([2, 2, 3])

        with pred_col1:
            risk_color = get_risk_color(risk_level)
            st.markdown(f"""
            <div class="kpi-card" style="--accent: {risk_color}">
                <div class="kpi-icon">{"🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"}</div>
                <div class="kpi-value" style="color:{risk_color}">{proba:.1%}</div>
                <div class="kpi-label">Attrition Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with pred_col2:
            st.markdown(f"""
            <div class="kpi-card" style="--accent: {risk_color}">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value" style="color:{risk_color}">{risk_level}</div>
                <div class="kpi-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

        with pred_col3:
            fig = create_gauge_chart(proba * 100, "Attrition Risk", color=risk_color)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📈 Feature Importance")
        imp_df = model_artifacts["importance_df"].head(15)

        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                    title=f"Top Features — {best_name}",
                    color="Importance",
                    color_continuous_scale=["#1B2838", "#00D4FF"])
        apply_chart_style(fig, height=500)
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Performance":
    render_section_header("Performance Analytics", "Comprehensive performance insights across the organization", "⭐")

    perf_df = load_performance_data()

    # Apply department filter if set
    dept_filter = st.session_state.get("filter_department", "All")
    if dept_filter != "All" and "Department" in perf_df.columns:
        perf_df = perf_df[perf_df["Department"] == dept_filter]

    role_filter = st.session_state.get("filter_job_role", "All")
    if role_filter != "All" and "Job Role" in perf_df.columns:
        perf_df = perf_df[perf_df["Job Role"] == role_filter]

    summary = compute_performance_summary(perf_df)

    # KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Avg Performance", f"{summary.get('avg_performance', 'N/A')}/100")
    with kpi2:
        st.metric("Avg KPI Score", f"{summary.get('avg_kpi', 'N/A')}")
    with kpi3:
        st.metric("Avg Attendance", f"{summary.get('avg_attendance', 'N/A')}%")
    with kpi4:
        st.metric("Avg Task Completion", f"{summary.get('avg_task_completion', 'N/A')}%")
    with kpi5:
        st.metric("Promotion Eligible", f"{summary.get('promo_eligible', 0)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    tab_a, tab_b, tab_c = st.tabs(["📊 Distribution", "🏢 Department Analysis", "🔗 Correlations"])

    with tab_a:
        ch1, ch2 = st.columns(2)
        with ch1:
            # Performance distribution
            perf_df_copy = perf_df.copy()
            perf_df_copy["Category"] = perf_df_copy["Performance Score"].apply(lambda x: classify_performance(x, "100"))
            cat_counts = perf_df_copy["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            color_map = {c: get_performance_color(c) for c in cat_counts["Category"]}
            fig = px.bar(cat_counts, x="Category", y="Count",
                        title="Performance Classification",
                        color="Category", color_discrete_map=color_map)
            apply_chart_style(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            # Performance score histogram
            fig = px.histogram(perf_df, x="Performance Score", nbins=25,
                             title="Performance Score Distribution",
                             color_discrete_sequence=[COLORS["accent_cyan"]])
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        dept_perf = get_department_performance(perf_df)
        if not dept_perf.empty:
            ch3, ch4 = st.columns(2)
            with ch3:
                fig = px.bar(dept_perf, x="Department", y="Performance Score",
                            title="Avg Performance by Department",
                            color="Performance Score",
                            color_continuous_scale=["#FF1744", "#FFB300", "#00C853"])
                apply_chart_style(fig)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with ch4:
                if "KPI Score" in dept_perf.columns:
                    fig = px.bar(dept_perf, x="Department", y="KPI Score",
                                title="Avg KPI Score by Department",
                                color="KPI Score",
                                color_continuous_scale=["#1B2838", "#00D4FF"])
                    apply_chart_style(fig)
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        ch5, ch6 = st.columns(2)
        with ch5:
            if "KPI Score" in perf_df.columns:
                fig = px.scatter(perf_df.sample(min(1000, len(perf_df)), random_state=42),
                               x="KPI Score", y="Performance Score",
                               title="KPI vs Performance Score",
                               color="Department" if "Department" in perf_df.columns else None,
                               color_discrete_sequence=COLORS["chart_colors"],
                               opacity=0.5)
                apply_chart_style(fig)
                st.plotly_chart(fig, use_container_width=True)

        with ch6:
            if "Training Hours" in perf_df.columns:
                fig = px.scatter(perf_df.sample(min(1000, len(perf_df)), random_state=42),
                               x="Training Hours", y="Performance Score",
                               title="Training Hours vs Performance",
                               color_discrete_sequence=[COLORS["accent_teal"]],
                               opacity=0.5)
                apply_chart_style(fig)
                st.plotly_chart(fig, use_container_width=True)

    # Employee performance profile
    st.markdown("---")
    render_section_header("Employee Performance Profile", icon="👤")
    emp_options = [f"{row['Employee ID']} — {row['Name']}" for _, row in perf_df.head(200).iterrows()]
    if emp_options:
        selected = st.selectbox("Select Employee", emp_options, key="perf_emp_select")
        sel_id = int(selected.split(" — ")[0])
        row = perf_df[perf_df["Employee ID"] == sel_id].iloc[0]

        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        with pcol1:
            st.metric("Performance", f"{row['Performance Score']}/100")
        with pcol2:
            st.metric("KPI", f"{round(row['KPI Score'], 1)}")
        with pcol3:
            st.metric("Attendance", f"{round(row['Attendance (%)'], 1)}%")
        with pcol4:
            st.metric("Task Completion", f"{round(row['Task Completion (%)'], 1)}%")
        with pcol5:
            st.metric("Peer Rating", f"{row['Peer Rating']}/5")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SKILL GAP & CAREER READINESS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Skill Gap":
    render_section_header("Skill Gap & Career Readiness", "Analyze skill alignment and career readiness for target roles", "🎯")

    occ_df = load_occupation_data()
    essential_df = load_essential_skills()
    software_df = load_software_skills()
    perf_pro = load_performance_pro_data()

    # Step 1: Select employee
    st.markdown("### Step 1: Select Employee / Current Role")
    employees = {row["EmployeeID"]: f"{row['EmployeeID']} — {row['Name']} ({row['JobRole']})"
                 for _, row in perf_pro.iterrows()}
    emp_ids = sorted(employees.keys())
    emp_labels = [employees[eid] for eid in emp_ids]

    default_idx = 0
    if st.session_state.get("selected_employee_id") in emp_ids:
        default_idx = emp_ids.index(st.session_state["selected_employee_id"])

    selected_label = st.selectbox("Employee", emp_labels, index=default_idx, key="sg_emp")
    selected_id = emp_ids[emp_labels.index(selected_label)]
    st.session_state["selected_employee_id"] = selected_id

    emp = perf_pro[perf_pro["EmployeeID"] == selected_id].iloc[0]
    current_role = emp["JobRole"]

    st.info(f"**Current Role:** {current_role} — {emp['Department']}")

    # Step 2: Select target occupation
    st.markdown("### Step 2: Select Target Occupation")
    occ_titles = get_occupation_titles(occ_df)

    # Default target
    default_target_idx = 0
    if st.session_state.get("selected_target_occupation") in occ_titles:
        default_target_idx = occ_titles.index(st.session_state["selected_target_occupation"])

    target_occ = st.selectbox("Target Occupation", occ_titles, index=default_target_idx, key="sg_target")
    st.session_state["selected_target_occupation"] = target_occ

    target_info = find_occupation_by_title(occ_df, target_occ)

    if target_info:
        with st.expander("📄 Occupation Description"):
            st.write(target_info.get("Description", "No description available."))

    # Steps 3-5: Compute skill gap (recomputes on every target change)
    with st.spinner("Analyzing skills..."):
        soc_code = target_info["O*NET-SOC Code"] if target_info else None

        # Get required skills
        required_skills = []
        if soc_code:
            required_skills = get_combined_skills_for_occupation(essential_df, software_df, soc_code)

        # Get current skills (inferred)
        current_skills, current_occ_title, is_inferred = infer_employee_skills(
            current_role, occ_df, essential_df, software_df
        )

        # Match
        matched, gaps, match_details = semantic_match_skills(current_skills, required_skills)
        readiness = calculate_readiness_score(matched, len(required_skills))

    # Display Results
    st.markdown("### Step 3–5: Skill Gap Analysis Results")

    if is_inferred:
        st.caption("ℹ️ **Inferred Current Skills** — Current skills are inferred from the employee's job role mapping to O*NET occupations. This is a prototype approximation, not actual verified employee skill data.")

    # Readiness Score
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        fig = create_gauge_chart(readiness, "Career Readiness")
        st.plotly_chart(fig, use_container_width=True)
    with res_col2:
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Analysis Summary</h4>
            <p><strong>Current Role:</strong> {current_role} → <em>{current_occ_title}</em></p>
            <p><strong>Target Role:</strong> {target_occ}</p>
            <p><strong>Total Required Skills:</strong> {len(required_skills)}</p>
            <p><strong>Matched Skills:</strong> <span style="color:#00C853">{len(matched)}</span></p>
            <p><strong>Skill Gaps:</strong> <span style="color:#FF1744">{len(gaps)}</span></p>
            <p><strong>Readiness Score:</strong> <span style="font-size:1.3rem; font-weight:700; color:{'#00C853' if readiness >= 60 else '#FFB300' if readiness >= 40 else '#FF1744'}">{readiness}%</span></p>
        </div>
        """, unsafe_allow_html=True)

    # Matched vs Gap columns
    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:
        st.markdown("#### ✅ Matched Skills")
        for s in matched[:20]:
            st.markdown(f'<div class="skill-match">✓ {s}</div>', unsafe_allow_html=True)
        if not matched:
            st.caption("No matched skills found.")

    with skill_col2:
        st.markdown("#### ❌ Skill Gaps")
        for s in gaps[:20]:
            st.markdown(f'<div class="skill-gap">✗ {s}</div>', unsafe_allow_html=True)
        if not gaps:
            st.caption("No gaps — excellent match! 🎉")

    # Match details
    if match_details:
        with st.expander("🔍 Matching Details"):
            details_df = pd.DataFrame(match_details)
            st.dataframe(details_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LEARNING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Recommendations":
    render_section_header("Learning Recommendations", "Personalized learning paths based on skill gap analysis", "📚")

    # Get skill gaps from session state (carry over from Skill Gap page)
    occ_df = load_occupation_data()
    essential_df = load_essential_skills()
    software_df = load_software_skills()
    perf_pro = load_performance_pro_data()

    # Employee & target from session
    selected_id = st.session_state.get("selected_employee_id")
    target_occ = st.session_state.get("selected_target_occupation")

    if selected_id and target_occ:
        emp_match = perf_pro[perf_pro["EmployeeID"] == selected_id]
        if not emp_match.empty:
            emp = emp_match.iloc[0]
            current_role = emp["JobRole"]
            st.info(f"**Employee:** {emp['Name']} (ID: {selected_id}) | **Current:** {current_role} | **Target:** {target_occ}")

            # Compute gaps
            current_skills, _, _ = infer_employee_skills(current_role, occ_df, essential_df, software_df)
            target_info = find_occupation_by_title(occ_df, target_occ)
            if target_info:
                required_skills = get_combined_skills_for_occupation(
                    essential_df, software_df, target_info["O*NET-SOC Code"]
                )
                matched, gaps, _ = semantic_match_skills(current_skills, required_skills)
            else:
                gaps = []
        else:
            gaps = []
            st.warning("Selected employee not found.")
    else:
        gaps = []
        st.info("💡 Visit the **Skill Gap** page first to select an employee and target role, or select below.")
        # Fallback: manual gap entry
        manual_gaps = st.text_input("Or enter skill gaps manually (comma-separated):",
                                    placeholder="e.g., Python, Machine Learning, Docker")
        if manual_gaps:
            gaps = [g.strip() for g in manual_gaps.split(",") if g.strip()]

    if gaps:
        # Generate recommendations
        recommendations = recommend_courses_for_gaps(gaps)
        learning_path = generate_learning_path(recommendations)
        certs = get_certification_suggestions(gaps)

        # Learning Path
        st.markdown("### 📖 Recommended Learning Path")

        for item in learning_path:
            phase_color = {"Foundation": "#00C853", "Core Development": "#00D4FF", "Advanced Specialization": "#7C4DFF"}.get(item["phase"], "#8899AA")
            st.markdown(f"""
            <div class="info-card" style="border-left: 3px solid {phase_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:{phase_color}; font-size:0.72rem; text-transform:uppercase; font-weight:600;">{item['phase']} — Priority {item['priority']}</span>
                        <h3 style="margin:4px 0; font-size:1rem;">{item['course_name']}</h3>
                        <p style="color:#6B7D8E; font-size:0.85rem; margin:2px 0;">🎯 For skill gap: <strong>{item['skill_gap']}</strong></p>
                        <p style="color:#8899AA; font-size:0.82rem;">{item['description']}</p>
                    </div>
                    <div style="text-align:right; min-width:120px;">
                        <span class="status-pill {'active' if item['difficulty']=='Beginner' else 'info' if item['difficulty']=='Intermediate' else 'warning'}">{item['difficulty']}</span>
                        <p style="color:#6B7D8E; font-size:0.82rem; margin-top:6px;">⏱ {item['duration_hours']} hours</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Certifications
        if certs:
            st.markdown("### 🎓 Suggested Certifications")
            for c in certs:
                st.markdown(f"- **{c['certification']}** — for _{c['skill']}_")

        # Summary Table
        with st.expander("📋 Full Recommendation Table"):
            rec_df = pd.DataFrame(recommendations)[
                ["skill_gap", "course_name", "difficulty", "duration_hours", "category", "reason"]
            ]
            rec_df.columns = ["Skill Gap", "Course", "Difficulty", "Hours", "Category", "Reason"]
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="info-card">
            <h4>💡 No Skill Gaps Identified</h4>
            <p>Navigate to the <strong>Skill Gap</strong> page to analyze an employee's skill alignment with a target role, or enter gaps manually above.</p>
        </div>
        """, unsafe_allow_html=True)

    # Course Catalog
    with st.expander("📚 Full Course Catalog"):
        catalog = get_course_catalog()
        st.dataframe(catalog, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — WORKFORCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ Workforce":
    render_section_header("Workforce Intelligence", "Organization-level skill analytics & strategic planning", "🏢")

    occ_df = load_occupation_data()
    essential_df = load_essential_skills()
    software_df = load_software_skills()
    perf_pro = load_performance_pro_data()
    perf_df = load_performance_data()

    tab_w1, tab_w2, tab_w3 = st.tabs(["📊 Skill Demand", "🔥 Technology Trends", "🔄 Hire vs Reskill"])

    with tab_w1:
        st.markdown("### Skill Demand Analysis")
        st.caption("Select target occupations to analyze required vs available skills across the organization.")

        occ_titles = get_occupation_titles(occ_df)
        selected_occs = st.multiselect("Select Target Occupations",
                                       occ_titles[:100],
                                       default=occ_titles[:3] if len(occ_titles) >= 3 else occ_titles[:1],
                                       key="wf_occs")

        if selected_occs:
            # Aggregate required skills
            all_required = {}
            for occ in selected_occs:
                info = find_occupation_by_title(occ_df, occ)
                if info:
                    skills = get_combined_skills_for_occupation(
                        essential_df, software_df, info["O*NET-SOC Code"]
                    )
                    for s in skills[:20]:
                        all_required[s] = all_required.get(s, 0) + 1

            # Estimate available (from employee roles)
            emp_roles = perf_pro["JobRole"].unique().tolist()
            available_skills = set()
            for role in emp_roles:
                skills, _, _ = infer_employee_skills(role, occ_df, essential_df, software_df)
                available_skills.update(skills[:10])

            # Build comparison
            skill_demand = []
            for skill, demand in sorted(all_required.items(), key=lambda x: -x[1])[:20]:
                is_available = any(
                    skill.lower() in av.lower() or av.lower() in skill.lower()
                    for av in available_skills
                )
                skill_demand.append({
                    "Skill": skill,
                    "Demand": demand,
                    "Available": "Yes" if is_available else "No",
                    "Gap": "No" if is_available else "Yes",
                })

            sd_df = pd.DataFrame(skill_demand)

            fig = px.bar(sd_df, x="Demand", y="Skill", orientation="h",
                        color="Gap",
                        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["positive"]},
                        title="Skill Demand & Gap Overview")
            apply_chart_style(fig, height=500)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            # Skill heatmap by occupation
            if len(selected_occs) > 1:
                st.markdown("### Skill Heatmap by Occupation")
                heatmap_data = []
                all_skills_set = list(all_required.keys())[:15]
                for occ in selected_occs:
                    info = find_occupation_by_title(occ_df, occ)
                    if info:
                        skills = get_combined_skills_for_occupation(
                            essential_df, software_df, info["O*NET-SOC Code"]
                        )
                        skills_lower = [s.lower() for s in skills]
                        for sk in all_skills_set:
                            has = 1 if sk.lower() in skills_lower else 0
                            heatmap_data.append({"Occupation": occ[:30], "Skill": sk, "Required": has})

                if heatmap_data:
                    hm_df = pd.DataFrame(heatmap_data)
                    pivot = hm_df.pivot(index="Occupation", columns="Skill", values="Required").fillna(0)
                    fig = px.imshow(pivot, title="Skill Requirements Heatmap",
                                  color_continuous_scale=["#1B2838", "#00D4FF"],
                                  aspect="auto")
                    apply_chart_style(fig, height=400)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_w2:
        st.markdown("### 🔥 Most In-Demand Technologies")

        hot_tech = software_df[software_df["Hot Technology"] == "Y"]
        tech_counts = hot_tech["Workplace Example"].value_counts().head(20).reset_index()
        tech_counts.columns = ["Technology", "Occupations Using"]

        fig = px.bar(tech_counts, x="Occupations Using", y="Technology", orientation="h",
                    title="Top 20 Hot Technologies (by # of occupations)",
                    color="Occupations Using",
                    color_continuous_scale=["#1B2838", "#FF6D00"])
        apply_chart_style(fig, height=600)
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # In-demand
        in_demand = software_df[software_df["In Demand"] == "Y"]
        id_counts = in_demand["Workplace Example"].value_counts().head(15).reset_index()
        id_counts.columns = ["Technology", "Count"]

        fig2 = px.bar(id_counts, x="Count", y="Technology", orientation="h",
                     title="Top In-Demand Technologies",
                     color="Count",
                     color_continuous_scale=["#1B2838", "#00C853"])
        apply_chart_style(fig2, height=450)
        fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_w3:
        st.markdown("### 🔄 Hire vs Reskill Analysis")
        st.caption("Estimate internal reskilling potential vs external hiring requirements for a target workforce need.")

        hr_col1, hr_col2 = st.columns(2)

        with hr_col1:
            target_for_hire = st.selectbox("Target Occupation", occ_titles[:100], key="hire_target")
            required_headcount = st.slider("Required Headcount", 10, 500, 100, key="hire_headcount")

        with hr_col2:
            reskill_threshold = st.slider(
                "Reskilling Eligibility Threshold (%)",
                min_value=20, max_value=80, value=40, step=5,
                help="Minimum readiness score for an employee to be considered reskillable",
                key="hire_threshold",
            )

        # Compute reskilling potential
        target_info = find_occupation_by_title(occ_df, target_for_hire)
        if target_info:
            target_skills = get_combined_skills_for_occupation(
                essential_df, software_df, target_info["O*NET-SOC Code"]
            )

            reskillable = 0
            emp_readiness = []
            for _, emp in perf_pro.iterrows():
                curr_skills, _, _ = infer_employee_skills(
                    emp["JobRole"], occ_df, essential_df, software_df
                )
                matched, gaps, _ = semantic_match_skills(curr_skills, target_skills, threshold=0.5)
                readiness = calculate_readiness_score(matched, len(target_skills))
                emp_readiness.append(readiness)
                if readiness >= reskill_threshold:
                    reskillable += 1

            # Scale to org size
            scale_factor = max(len(perf_pro) / 100, 1)
            est_reskillable = min(reskillable, required_headcount)
            external_needed = max(required_headcount - est_reskillable, 0)

            st.markdown("---")

            hr1, hr2, hr3 = st.columns(3)
            with hr1:
                render_kpi_card("🎯", f"{required_headcount}", "Required Headcount",
                               accent_color=COLORS["accent_cyan"])
            with hr2:
                render_kpi_card("🔄", f"{est_reskillable}", "Potentially Reskillable",
                               delta=f"{reskill_threshold}% threshold",
                               delta_type="positive",
                               accent_color=COLORS["positive"])
            with hr3:
                render_kpi_card("🆕", f"{external_needed}", "External Hiring Needed",
                               accent_color=COLORS["warning"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualization
            fig = go.Figure(data=[
                go.Bar(name="Reskillable", x=["Workforce Plan"], y=[est_reskillable],
                      marker_color=COLORS["positive"]),
                go.Bar(name="External Hire", x=["Workforce Plan"], y=[external_needed],
                      marker_color=COLORS["warning"]),
            ])
            fig.update_layout(barmode="stack", title="Hire vs Reskill Breakdown")
            apply_chart_style(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Readiness distribution
            fig2 = px.histogram(x=emp_readiness, nbins=20,
                              title="Employee Readiness Distribution for Target Role",
                              labels={"x": "Readiness Score (%)", "count": "Employees"},
                              color_discrete_sequence=[COLORS["accent_cyan"]])
            fig2.add_vline(x=reskill_threshold, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text=f"Threshold: {reskill_threshold}%")
            apply_chart_style(fig2, height=350)
            st.plotly_chart(fig2, use_container_width=True)

            st.caption("⚠️ This is an illustrative prototype analysis. Reskilling estimates are based on inferred skills from job roles and O*NET occupational data.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — AI HR ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ AI Assistant":
    render_section_header("AI HR Assistant", "Intelligent workforce assistant powered by agentic AI", "🤖")

    st.markdown("""
    <div class="info-card">
        <h4>💡 What can I help with?</h4>
        <p style="color:#8899AA;">
        Ask me about HR policies, employee information, attrition risk, required skills for roles,
        skill gaps, course recommendations, performance data, or workforce statistics.
        </p>
        <p style="color:#5A6B7C; font-size:0.8rem;">
        Try: "What is the leave policy?" • "What skills does a Data Scientist need?" •
        "How can Employee 1 become a Software Developer?" • "What are the workforce statistics?"
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg:
                with st.expander("🔍 Reasoning Trace"):
                    for step in msg["trace"]:
                        icon = step.get("icon", "🔧")
                        status_icon = "✅" if step["status"] == "completed" else "❌"
                        st.markdown(f"{status_icon} {icon} **{step['description']}**")

    # Chat input
    user_input = st.chat_input("Ask me anything about HR, skills, careers, or policies...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                tools_to_invoke = detect_intent(user_input)
                results, trace = execute_tools(user_input, tools_to_invoke)
                response = generate_response(user_input, results, trace)

            # Show reasoning trace
            with st.expander("🔍 Reasoning Trace"):
                for step in trace:
                    icon = step.get("icon", "🔧")
                    status_icon = "✅" if step["status"] == "completed" else "❌"
                    st.markdown(f"{status_icon} {icon} **{step['description']}**")
                st.caption(f"Tools invoked: {' → '.join([t['tool'] for t in trace])}")

            # Stream response
            placeholder = st.empty()
            stream_response(response, placeholder)

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "trace": trace,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "◉ About":
    render_section_header("About This Project", "Architecture, methodology & technical details", "ℹ️")

    st.markdown("""
    ### 🧠 Agentic HRMS — Workforce Intelligence Platform

    An AI-powered HR analytics platform that demonstrates end-to-end workforce intelligence:
    from employee analytics to attrition prediction, skill-gap analysis, career readiness,
    and personalized learning recommendations.

    ---

    ### 🏗️ Architecture

    ```
    Employee Data → Analysis → Prediction → Skill Gap → Career Readiness → Recommendations → HR Decision Support
    ```
    """)

    st.markdown("""
    ### 📊 Data Pipeline

    | Dataset | Records | Purpose |
    |---------|---------|---------|
    | HR Data Analysis | 2,845 | Employee demographics, engagement, training |
    | Employee Attrition | 1,470 | ML training for attrition prediction |
    | Performance Dataset | 5,000 | KPIs, attendance, task completion |
    | Performance Pro | 500 | Extended profiles with salary, attrition risk |
    | Occupation Data | 1,016 | O*NET occupations and descriptions |
    | Essential Skills | 18,200 | Core skills per occupation (importance/level) |
    | Software Skills | 31,821 | Technology/software per occupation |

    ---

    ### 🤖 ML Models

    **Attrition Prediction:**
    - Logistic Regression & Random Forest (compared by F1 score)
    - Class-balanced training for imbalanced target (16% attrition rate)
    - Feature importance from the best model

    **Skill Matching:**
    - Sentence-Transformers (all-MiniLM-L6-v2) for semantic similarity
    - Fuzzy string matching (SequenceMatcher) as fallback
    - Combined scoring with configurable threshold

    ---

    ### 🔧 Agentic Workflow

    The AI assistant uses a lightweight orchestrator with tool-based routing:
    """)

    tools_data = pd.DataFrame([
        {"Tool": name, "Description": info["description"], "Icon": info["icon"]}
        for name, info in TOOLS.items()
    ])
    st.dataframe(tools_data, use_container_width=True, hide_index=True)

    st.markdown("""
    ---

    ### 📚 RAG Pipeline

    - **Documents:** 5 sample HR policy documents
    - **Chunking:** Section-based with overlap
    - **Embeddings:** TF-IDF vectorization
    - **Retrieval:** Cosine similarity search
    - **Response:** Structured policy extraction (no LLM fabrication)

    ---

    ### ⚡ Technology Stack

    | Component | Technology |
    |-----------|-----------|
    | Frontend | Streamlit |
    | Charts | Plotly |
    | ML | scikit-learn |
    | NLP | Sentence-Transformers |
    | Data | pandas, NumPy |
    | RAG | TF-IDF + Cosine Similarity |

    ---

    ### ⚠️ Limitations

    - This is a **prototype** for demonstration purposes
    - Employee skills are **inferred** from job roles, not actual verified skill data
    - Attrition predictions are based on a specific dataset and may not generalize
    - HR policies are **sample documents** for RAG demonstration
    - The "Live Mode" shows **simulated** metric perturbation, not real-time data
    - No external LLM API is used; responses are rule-based + retrieval-based
    """)

    # Status indicators
    st.markdown("---")
    st.markdown("### System Status")
    status_cols = st.columns(4)
    with status_cols[0]:
        st.markdown(render_status_pill("● Data Connected", "active"), unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown(render_status_pill("● Model Ready", "active"), unsafe_allow_html=True)
    with status_cols[2]:
        st.markdown(render_status_pill("● RAG Ready", "active"), unsafe_allow_html=True)
    with status_cols[3]:
        lm = "active" if st.session_state.get("live_mode") else "info"
        st.markdown(render_status_pill(f"● Live Mode: {'ON' if st.session_state.get('live_mode') else 'OFF'}", lm), unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
render_footer()
