"""
Agentic HRMS — Lightweight Agentic Orchestrator
Routes user queries to appropriate tools and generates responses with visible reasoning traces.
"""
import streamlit as st
import re
import time
import pandas as pd

from src.data_loader import (
    load_hr_data, load_attrition_data, load_performance_data,
    load_performance_pro_data, load_occupation_data,
    load_essential_skills, load_software_skills,
)
from src.skill_gap import (
    get_occupation_titles, find_occupation_by_title,
    get_combined_skills_for_occupation, infer_employee_skills,
    semantic_match_skills, calculate_readiness_score,
    map_role_to_occupation,
)
from src.recommender import recommend_courses_for_gaps, get_certification_suggestions
from src.rag import build_rag_index, search_policies, format_rag_response
from src.performance import classify_performance


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOLS = {
    "get_employee_profile": {
        "description": "Retrieve employee profile information",
        "icon": "👤",
    },
    "predict_attrition": {
        "description": "Predict employee attrition risk",
        "icon": "⚠️",
    },
    "get_required_skills": {
        "description": "Get required skills for a target occupation",
        "icon": "🎯",
    },
    "calculate_skill_gap": {
        "description": "Calculate skill gap between current and target role",
        "icon": "📊",
    },
    "calculate_readiness": {
        "description": "Calculate career readiness score",
        "icon": "📈",
    },
    "recommend_courses": {
        "description": "Recommend courses for skill gaps",
        "icon": "📚",
    },
    "search_hr_policy": {
        "description": "Search HR policy documents",
        "icon": "📋",
    },
    "get_performance_info": {
        "description": "Retrieve employee performance data",
        "icon": "⭐",
    },
    "get_workforce_stats": {
        "description": "Get workforce statistics and analytics",
        "icon": "📉",
    },
}


# ─── Intent Detection ─────────────────────────────────────────────────────────

def detect_intent(query):
    """
    Detect user intent and determine which tools to invoke.
    Returns list of tool names in execution order.
    """
    query_lower = query.lower()

    tools_to_invoke = []

    # Policy / HR knowledge queries
    policy_keywords = [
        "policy", "leave", "holiday", "vacation", "sick", "maternity",
        "training policy", "promotion", "work from home", "wfh", "remote",
        "handbook", "code of conduct", "benefits", "salary", "compensation",
        "grievance", "exit", "notice period", "insurance",
    ]
    if any(kw in query_lower for kw in policy_keywords):
        tools_to_invoke.append("search_hr_policy")
        return tools_to_invoke

    # Employee profile queries
    employee_keywords = ["employee", "profile", "who is", "tell me about", "information about"]
    has_employee = any(kw in query_lower for kw in employee_keywords)

    # Attrition queries
    attrition_keywords = ["attrition", "quit", "leave the company", "turnover", "resign", "retention", "risk"]
    has_attrition = any(kw in query_lower for kw in attrition_keywords)

    # Skill / career queries
    skill_keywords = ["skill", "gap", "career", "readiness", "become", "transition", "path", "role"]
    has_skill = any(kw in query_lower for kw in skill_keywords)

    # Course / learning queries
    course_keywords = ["course", "learn", "training", "recommend", "study", "certification", "upskill"]
    has_course = any(kw in query_lower for kw in course_keywords)

    # Performance queries
    perf_keywords = ["performance", "kpi", "rating", "attendance", "task completion"]
    has_perf = any(kw in query_lower for kw in perf_keywords)

    # Workforce / stats queries
    workforce_keywords = ["workforce", "organization", "department", "statistics", "how many", "total", "average", "headcount"]
    has_workforce = any(kw in query_lower for kw in workforce_keywords)

    # Occupation / required skills queries
    occupation_keywords = ["required for", "need for", "skills for", "requirements for", "what does a", "data scientist", "software", "engineer", "manager", "analyst"]
    has_occupation = any(kw in query_lower for kw in occupation_keywords)

    # Complex career path query (e.g., "How can Employee X become a Data Scientist?")
    if has_employee and (has_skill or "become" in query_lower):
        tools_to_invoke = [
            "get_employee_profile",
            "get_required_skills",
            "calculate_skill_gap",
            "calculate_readiness",
            "recommend_courses",
        ]
        return tools_to_invoke

    # Build tool list based on detected intents
    if has_employee:
        tools_to_invoke.append("get_employee_profile")
    if has_attrition:
        tools_to_invoke.append("predict_attrition")
    if has_occupation or (has_skill and not has_course):
        tools_to_invoke.append("get_required_skills")
    if has_skill and not has_course:
        tools_to_invoke.append("calculate_skill_gap")
        tools_to_invoke.append("calculate_readiness")
    if has_course:
        if has_skill:
            tools_to_invoke.append("get_required_skills")
            tools_to_invoke.append("calculate_skill_gap")
        tools_to_invoke.append("recommend_courses")
    if has_perf:
        tools_to_invoke.append("get_performance_info")
    if has_workforce:
        tools_to_invoke.append("get_workforce_stats")

    # Default to workforce stats + policy search if nothing matched
    if not tools_to_invoke:
        tools_to_invoke = ["search_hr_policy", "get_workforce_stats"]

    return tools_to_invoke


# ─── Tool Execution ───────────────────────────────────────────────────────────

def extract_employee_id(query):
    """Try to extract an employee ID from the query."""
    patterns = [
        r'employee\s*(?:id\s*)?#?\s*(\d+)',
        r'emp\s*(?:id\s*)?#?\s*(\d+)',
        r'id\s*(\d+)',
        r'#(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_occupation(query):
    """Try to extract a target occupation from the query."""
    occupations = get_occupation_titles(load_occupation_data())
    query_lower = query.lower()

    # Common role names
    role_keywords = {
        "data scientist": "Data Scientists",
        "software developer": "Software Developers",
        "software engineer": "Software Developers",
        "project manager": "Project Management Specialists",
        "hr manager": "Human Resources Managers",
        "financial analyst": "Financial Analysts",
        "marketing manager": "Marketing Managers",
        "sales manager": "Sales Managers",
        "data analyst": "Data Scientists",
        "cybersecurity": "Information Security Analysts",
        "web developer": "Web Developers",
    }

    for keyword, occ_name in role_keywords.items():
        if keyword in query_lower:
            # Find the actual occupation in our data
            for occ in occupations:
                if occ_name.lower() in occ.lower() or occ.lower() in occ_name.lower():
                    return occ
            # Return first partial match
            for occ in occupations:
                if keyword.split()[0] in occ.lower():
                    return occ

    return None


def execute_tools(query, tools_to_invoke):
    """
    Execute the identified tools and collect results.
    Returns dict of tool results and reasoning trace.
    """
    results = {}
    trace = []

    # Extract context from query
    emp_id = extract_employee_id(query)
    target_occ = extract_occupation(query)

    # Check session state for employee selection
    if emp_id is None and st.session_state.get("selected_employee_id"):
        emp_id = st.session_state["selected_employee_id"]

    for tool_name in tools_to_invoke:
        tool_info = TOOLS.get(tool_name, {})
        trace.append({
            "tool": tool_name,
            "icon": tool_info.get("icon", "🔧"),
            "description": tool_info.get("description", tool_name),
            "status": "running",
        })

        try:
            if tool_name == "get_employee_profile":
                result = _execute_get_profile(emp_id)
            elif tool_name == "predict_attrition":
                result = _execute_predict_attrition(emp_id)
            elif tool_name == "get_required_skills":
                result = _execute_get_required_skills(target_occ)
            elif tool_name == "calculate_skill_gap":
                result = _execute_calculate_skill_gap(emp_id, target_occ)
            elif tool_name == "calculate_readiness":
                result = _execute_calculate_readiness(emp_id, target_occ)
            elif tool_name == "recommend_courses":
                result = _execute_recommend_courses(emp_id, target_occ)
            elif tool_name == "search_hr_policy":
                result = _execute_search_policy(query)
            elif tool_name == "get_performance_info":
                result = _execute_get_performance(emp_id)
            elif tool_name == "get_workforce_stats":
                result = _execute_get_workforce_stats()
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            results[tool_name] = result
            trace[-1]["status"] = "completed"
        except Exception as e:
            results[tool_name] = {"error": str(e)}
            trace[-1]["status"] = "error"
            trace[-1]["error"] = str(e)

    return results, trace


def _execute_get_profile(emp_id):
    """Get employee profile from available datasets."""
    if emp_id is None:
        return {"message": "No employee ID specified. Please mention an employee ID in your question."}

    # Search across datasets
    for loader, id_col, name in [
        (load_performance_pro_data, "EmployeeID", "Performance Pro"),
        (load_hr_data, "Employee ID", "HR Data"),
        (load_performance_data, "Employee ID", "Performance Data"),
    ]:
        df = loader()
        if id_col in df.columns:
            match = df[df[id_col] == emp_id]
            if not match.empty:
                profile = match.iloc[0].to_dict()
                profile["_source"] = name
                return profile

    return {"message": f"Employee ID {emp_id} not found in any dataset."}


def _execute_predict_attrition(emp_id):
    """Get attrition risk information."""
    if emp_id is None:
        # Return general attrition statistics
        df = load_attrition_data()
        attr_counts = df["Attrition"].value_counts().to_dict()
        rate = round(attr_counts.get("Yes", 0) / len(df) * 100, 1)
        return {
            "type": "general",
            "attrition_rate": f"{rate}%",
            "total_employees": len(df),
            "left": attr_counts.get("Yes", 0),
            "stayed": attr_counts.get("No", 0),
        }

    # Check performance_pro for AttritionRisk
    df = load_performance_pro_data()
    match = df[df["EmployeeID"] == emp_id]
    if not match.empty:
        risk = match.iloc[0].get("AttritionRisk", "Unknown")
        return {
            "type": "individual",
            "employee_id": emp_id,
            "attrition_risk": risk,
            "name": match.iloc[0].get("Name", "Unknown"),
        }

    return {"message": f"No attrition data found for Employee ID {emp_id}."}


def _execute_get_required_skills(target_occ):
    """Get required skills for a target occupation."""
    occ_df = load_occupation_data()
    essential_df = load_essential_skills()
    software_df = load_software_skills()

    if target_occ is None:
        # Return popular occupations
        return {
            "message": "No specific occupation mentioned. Here are some popular target roles you can ask about.",
            "sample_roles": get_occupation_titles(occ_df)[:10],
        }

    occ = find_occupation_by_title(occ_df, target_occ)
    if occ is None:
        return {"message": f"Occupation '{target_occ}' not found."}

    soc_code = occ["O*NET-SOC Code"]
    skills = get_combined_skills_for_occupation(essential_df, software_df, soc_code)

    return {
        "occupation": target_occ,
        "soc_code": soc_code,
        "description": occ.get("Description", ""),
        "required_skills": skills[:25],
        "total_skills": len(skills),
    }


def _execute_calculate_skill_gap(emp_id, target_occ):
    """Calculate skill gap for an employee targeting an occupation."""
    occ_df = load_occupation_data()
    essential_df = load_essential_skills()
    software_df = load_software_skills()

    # Get employee's current role
    current_role = "Unknown"
    if emp_id is not None:
        for loader, id_col, role_col in [
            (load_performance_pro_data, "EmployeeID", "JobRole"),
            (load_performance_data, "Employee ID", "Job Role"),
            (load_hr_data, "Employee ID", "Title"),
        ]:
            df = loader()
            if id_col in df.columns and role_col in df.columns:
                match = df[df[id_col] == emp_id]
                if not match.empty:
                    current_role = match.iloc[0][role_col]
                    break

    # Get current skills (inferred from role)
    current_skills, current_occ, is_inferred = infer_employee_skills(
        current_role, occ_df, essential_df, software_df
    )

    if target_occ is None:
        return {
            "current_role": current_role,
            "current_skills": current_skills[:15],
            "message": "No target occupation specified for gap analysis.",
        }

    # Get target skills
    target_occ_info = find_occupation_by_title(occ_df, target_occ)
    if target_occ_info is None:
        return {"message": f"Target occupation '{target_occ}' not found."}

    target_skills = get_combined_skills_for_occupation(
        essential_df, software_df, target_occ_info["O*NET-SOC Code"]
    )

    # Compute gap
    matched, gaps, details = semantic_match_skills(current_skills, target_skills)

    return {
        "current_role": current_role,
        "target_role": target_occ,
        "matched_skills": matched,
        "skill_gaps": gaps,
        "total_required": len(target_skills),
        "is_inferred": is_inferred,
    }


def _execute_calculate_readiness(emp_id, target_occ):
    """Calculate readiness score."""
    gap_result = _execute_calculate_skill_gap(emp_id, target_occ)
    if "message" in gap_result and "skill_gaps" not in gap_result:
        return gap_result

    matched = gap_result.get("matched_skills", [])
    total = gap_result.get("total_required", 0)
    score = calculate_readiness_score(matched, total)

    return {
        "readiness_score": f"{score}%",
        "matched_count": len(matched),
        "total_required": total,
        "gap_count": len(gap_result.get("skill_gaps", [])),
    }


def _execute_recommend_courses(emp_id, target_occ):
    """Recommend courses based on skill gaps."""
    gap_result = _execute_calculate_skill_gap(emp_id, target_occ)
    gaps = gap_result.get("skill_gaps", [])

    if not gaps:
        return {"message": "No skill gaps identified. The employee appears well-prepared for the target role!"}

    recommendations = recommend_courses_for_gaps(gaps[:10])
    certs = get_certification_suggestions(gaps[:10])

    return {
        "skill_gaps_addressed": len(gaps),
        "recommendations": recommendations[:8],
        "certifications": certs[:5],
    }


def _execute_search_policy(query):
    """Search HR policy documents."""
    rag_index = build_rag_index()
    results = search_policies(query, rag_index, top_k=3)
    return {
        "type": "policy",
        "response": format_rag_response(query, results),
        "sources": [r["source"] for r in results],
    }


def _execute_get_performance(emp_id):
    """Get performance information."""
    if emp_id is not None:
        df = load_performance_data()
        match = df[df["Employee ID"] == emp_id]
        if not match.empty:
            row = match.iloc[0]
            score = row.get("Performance Score", 0)
            return {
                "type": "individual",
                "employee_id": emp_id,
                "name": row.get("Name", "Unknown"),
                "performance_score": score,
                "category": classify_performance(score),
                "kpi_score": row.get("KPI Score", "N/A"),
                "attendance": row.get("Attendance (%)", "N/A"),
            }

    # General performance stats
    df = load_performance_data()
    return {
        "type": "general",
        "avg_performance": round(df["Performance Score"].mean(), 1),
        "total_employees": len(df),
        "promotion_eligible": df[df["Promotion Eligibility"] == "Yes"].shape[0],
    }


def _execute_get_workforce_stats():
    """Get workforce statistics."""
    hr_df = load_hr_data()
    perf_df = load_performance_data()

    stats = {
        "total_employees_hr": len(hr_df),
        "active_employees": hr_df[hr_df["EmployeeStatus"] == "Active"].shape[0] if "EmployeeStatus" in hr_df.columns else "N/A",
        "departments": hr_df["DepartmentType"].nunique() if "DepartmentType" in hr_df.columns else "N/A",
        "avg_engagement": round(hr_df["Engagement Score"].mean(), 2) if "Engagement Score" in hr_df.columns else "N/A",
        "avg_satisfaction": round(hr_df["Satisfaction Score"].mean(), 2) if "Satisfaction Score" in hr_df.columns else "N/A",
        "performance_employees": len(perf_df),
        "avg_performance_score": round(perf_df["Performance Score"].mean(), 1) if "Performance Score" in perf_df.columns else "N/A",
    }
    return stats


# ─── Response Generation ──────────────────────────────────────────────────────

def generate_response(query, results, trace):
    """Generate a natural language response from tool results."""
    parts = []

    for tool_name, result in results.items():
        if isinstance(result, dict):
            if "error" in result:
                parts.append(f"⚠️ Error in {tool_name}: {result['error']}")
                continue

            if result.get("type") == "policy":
                parts.append(result.get("response", ""))
                continue

            if "message" in result and len(result) <= 3:
                parts.append(result["message"])
                continue

            # Format based on tool
            if tool_name == "get_employee_profile":
                name = result.get("Name", result.get("name", "Unknown"))
                dept = result.get("Department", result.get("DepartmentType", "N/A"))
                role = result.get("JobRole", result.get("Job Role", result.get("Title", "N/A")))
                parts.append(f"**Employee Profile:** {name}\n- Department: {dept}\n- Role: {role}")

                # Add more fields
                for key in ["Age", "PerformanceRating", "Performance Score", "YearsAtCompany", "MonthlySalary"]:
                    if key in result and result[key] is not None:
                        parts.append(f"- {key}: {result[key]}")

            elif tool_name == "predict_attrition":
                if result.get("type") == "individual":
                    parts.append(f"**Attrition Risk for {result.get('name', 'Employee')}:** {result.get('attrition_risk', 'Unknown')}")
                else:
                    parts.append(f"**Attrition Overview:**\n- Overall Rate: {result.get('attrition_rate', 'N/A')}\n- Employees who left: {result.get('left', 'N/A')}\n- Employees retained: {result.get('stayed', 'N/A')}")

            elif tool_name == "get_required_skills":
                skills = result.get("required_skills", [])
                occ = result.get("occupation", "Unknown")
                parts.append(f"**Skills Required for {occ}:**")
                for s in skills[:15]:
                    parts.append(f"  • {s}")
                if len(skills) > 15:
                    parts.append(f"  ... and {len(skills) - 15} more")

            elif tool_name == "calculate_skill_gap":
                matched = result.get("matched_skills", [])
                gaps = result.get("skill_gaps", [])
                parts.append(f"**Skill Gap Analysis ({result.get('current_role', '?')} → {result.get('target_role', '?')}):**")
                if result.get("is_inferred"):
                    parts.append("*Note: Current skills are inferred from job role.*")
                parts.append(f"\n✅ **Matched Skills ({len(matched)}):**")
                for s in matched[:10]:
                    parts.append(f"  ✓ {s}")
                parts.append(f"\n❌ **Skill Gaps ({len(gaps)}):**")
                for s in gaps[:10]:
                    parts.append(f"  ✗ {s}")

            elif tool_name == "calculate_readiness":
                parts.append(f"**Career Readiness Score:** {result.get('readiness_score', 'N/A')}")
                parts.append(f"- Matched: {result.get('matched_count', 0)} / {result.get('total_required', 0)} skills")

            elif tool_name == "recommend_courses":
                recs = result.get("recommendations", [])
                if recs:
                    parts.append("**📚 Recommended Courses:**")
                    for r in recs[:5]:
                        parts.append(f"  {r['course_name']} ({r['difficulty']}, {r['duration_hours']}h)")
                        parts.append(f"    ↳ For: {r['skill_gap']}")
                certs = result.get("certifications", [])
                if certs:
                    parts.append("\n**🎓 Suggested Certifications:**")
                    for c in certs:
                        parts.append(f"  • {c['certification']} (for {c['skill']})")

            elif tool_name == "get_performance_info":
                if result.get("type") == "individual":
                    parts.append(f"**Performance for {result.get('name', 'Employee')}:**")
                    parts.append(f"- Score: {result.get('performance_score', 'N/A')} ({result.get('category', '')})")
                    parts.append(f"- KPI: {result.get('kpi_score', 'N/A')}")
                    parts.append(f"- Attendance: {result.get('attendance', 'N/A')}%")
                else:
                    parts.append(f"**Performance Overview:**\n- Average Score: {result.get('avg_performance', 'N/A')}\n- Promotion Eligible: {result.get('promotion_eligible', 'N/A')}")

            elif tool_name == "get_workforce_stats":
                parts.append("**Workforce Statistics:**")
                parts.append(f"- Total Employees (HR): {result.get('total_employees_hr', 'N/A')}")
                parts.append(f"- Active: {result.get('active_employees', 'N/A')}")
                parts.append(f"- Departments: {result.get('departments', 'N/A')}")
                parts.append(f"- Avg Engagement: {result.get('avg_engagement', 'N/A')}")
                parts.append(f"- Avg Satisfaction: {result.get('avg_satisfaction', 'N/A')}")

    if not parts:
        return "I processed your query but couldn't generate a meaningful response. Could you please rephrase your question?"

    return "\n".join(parts)


def stream_response(response_text, placeholder):
    """Simulate streaming/typing effect for the response."""
    displayed = ""
    for char in response_text:
        displayed += char
        placeholder.markdown(displayed + "▌")
        # Variable speed for natural feel
        if char in ".!?\n":
            time.sleep(0.03)
        elif char == " ":
            time.sleep(0.008)
        else:
            time.sleep(0.004)
    placeholder.markdown(displayed)
