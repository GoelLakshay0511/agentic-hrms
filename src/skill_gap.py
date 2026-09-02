"""
Agentic HRMS — Skill Gap Analysis & Career Readiness
Uses O*NET occupation/skill data with semantic + fuzzy matching.
"""
import streamlit as st
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re


# ─── Semantic Matching Engine ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading skill matching model...")
def load_sentence_model():
    """Load sentence-transformers model for semantic skill matching."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception:
        return None


def normalize_skill(skill):
    """Normalize a skill string for comparison."""
    s = str(skill).lower().strip()
    s = re.sub(r'[^a-z0-9\s\+\#\.]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def fuzzy_match_score(s1, s2):
    """Compute fuzzy similarity between two skill strings."""
    n1 = normalize_skill(s1)
    n2 = normalize_skill(s2)
    if n1 == n2:
        return 1.0
    # Check containment
    if n1 in n2 or n2 in n1:
        return 0.85
    return SequenceMatcher(None, n1, n2).ratio()


def semantic_match_skills(current_skills, required_skills, threshold=0.55):
    """
    Match current skills against required skills using semantic similarity.
    Returns matched and gap lists.
    """
    model = load_sentence_model()

    matched = []
    gaps = []
    match_details = []

    if model is not None and len(current_skills) > 0 and len(required_skills) > 0:
        try:
            curr_embeddings = model.encode(current_skills, convert_to_numpy=True)
            req_embeddings = model.encode(required_skills, convert_to_numpy=True)

            # Compute cosine similarity matrix
            from numpy.linalg import norm
            curr_norm = curr_embeddings / (norm(curr_embeddings, axis=1, keepdims=True) + 1e-10)
            req_norm = req_embeddings / (norm(req_embeddings, axis=1, keepdims=True) + 1e-10)
            sim_matrix = curr_norm @ req_norm.T  # (n_curr, n_req)

            for j, req_skill in enumerate(required_skills):
                max_sim = sim_matrix[:, j].max() if sim_matrix.shape[0] > 0 else 0
                best_idx = sim_matrix[:, j].argmax() if sim_matrix.shape[0] > 0 else -1

                # Also check fuzzy match
                fuzzy_scores = [fuzzy_match_score(cs, req_skill) for cs in current_skills]
                best_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0
                best_fuzzy_idx = fuzzy_scores.index(best_fuzzy) if fuzzy_scores else -1

                effective_score = max(max_sim, best_fuzzy)
                if effective_score >= threshold:
                    best = current_skills[best_idx] if max_sim >= best_fuzzy else current_skills[best_fuzzy_idx]
                    matched.append(req_skill)
                    match_details.append({
                        "required": req_skill,
                        "matched_with": best,
                        "score": round(float(effective_score), 3),
                    })
                else:
                    gaps.append(req_skill)

            return matched, gaps, match_details
        except Exception:
            pass

    # Fallback: fuzzy matching only
    for req_skill in required_skills:
        best_score = 0
        best_match = None
        for curr_skill in current_skills:
            score = fuzzy_match_score(curr_skill, req_skill)
            if score > best_score:
                best_score = score
                best_match = curr_skill

        if best_score >= threshold:
            matched.append(req_skill)
            match_details.append({
                "required": req_skill,
                "matched_with": best_match,
                "score": round(best_score, 3),
            })
        else:
            gaps.append(req_skill)

    return matched, gaps, match_details


# ─── Occupation & Skill Extraction ────────────────────────────────────────────

def get_occupation_titles(occupation_df):
    """Get sorted list of all occupation titles."""
    return sorted(occupation_df["Title"].unique().tolist())


def find_occupation_by_title(occupation_df, title):
    """Find occupation details by title."""
    row = occupation_df[occupation_df["Title"] == title]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_essential_skills_for_occupation(essential_skills_df, soc_code):
    """Get essential skills for an occupation by O*NET-SOC Code."""
    skills = essential_skills_df[
        (essential_skills_df["O*NET-SOC Code"] == soc_code) &
        (essential_skills_df["Scale Name"] == "Importance") &
        (essential_skills_df["Recommend Suppress"] != "Y")
    ]
    return skills[["Element Name", "Data Value"]].drop_duplicates().sort_values(
        "Data Value", ascending=False
    ).reset_index(drop=True)


def get_software_skills_for_occupation(software_skills_df, soc_code):
    """Get software/technology skills for an occupation by O*NET-SOC Code."""
    skills = software_skills_df[software_skills_df["O*NET-SOC Code"] == soc_code]
    result = skills[["Workplace Example", "Element Name", "Hot Technology", "In Demand"]].drop_duplicates()
    return result.reset_index(drop=True)


def get_combined_skills_for_occupation(essential_df, software_df, soc_code):
    """Get combined essential + software skills for an occupation."""
    essential = get_essential_skills_for_occupation(essential_df, soc_code)
    essential_list = essential["Element Name"].tolist()

    software = get_software_skills_for_occupation(software_df, soc_code)
    # Use Element Name (category) + Workplace Example (specific tool)
    software_categories = software["Element Name"].unique().tolist()
    software_tools = software["Workplace Example"].unique().tolist()

    # Combine: essential skills + software categories + top tools
    hot_tools = software[software["Hot Technology"] == "Y"]["Workplace Example"].unique().tolist()
    in_demand = software[software["In Demand"] == "Y"]["Workplace Example"].unique().tolist()

    # Priority: hot tech + in-demand + top categories
    combined = essential_list + software_categories[:10] + hot_tools[:15] + in_demand[:10]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in combined:
        ns = normalize_skill(s)
        if ns not in seen:
            seen.add(ns)
            unique.append(s)

    return unique


# ─── Job Role to Occupation Mapping ───────────────────────────────────────────

# Pre-built mapping from common HR job roles to O*NET occupations
ROLE_TO_OCCUPATION_HINTS = {
    "sales executive": "Sales Representatives, Wholesale and Manufacturing",
    "sales representative": "Sales Representatives, Wholesale and Manufacturing",
    "area sales manager": "Sales Managers",
    "account manager": "Sales Managers",
    "marketing executive": "Market Research Analysts and Marketing Specialists",
    "seo specialist": "Search Marketing Strategists",
    "seo analyst": "Search Marketing Strategists",
    "content strategist": "Technical Writers",
    "content lead": "Technical Writers",
    "software engineer": "Software Developers",
    "developer": "Software Developers",
    "data analyst": "Data Scientists",
    "data scientist": "Data Scientists",
    "cybersecurity specialist": "Information Security Analysts",
    "hr manager": "Human Resources Managers",
    "hr executive": "Human Resources Specialists",
    "recruitment specialist": "Human Resources Specialists",
    "employee relations": "Human Resources Specialists",
    "accountant": "Accountants and Auditors",
    "auditor": "Accountants and Auditors",
    "financial analyst": "Financial Analysts",
    "production technician i": "Production Workers, All Other",
    "production technician ii": "Production Workers, All Other",
    "production manager": "Industrial Production Managers",
    "business development": "Market Research Analysts and Marketing Specialists",
    "research scientist": "Natural Sciences Managers",
    "laboratory technician": "Chemical Technicians",
    "manufacturing director": "Industrial Production Managers",
    "healthcare representative": "Sales Representatives, Wholesale and Manufacturing",
    "manager": "General and Operations Managers",
    "research director": "Natural Sciences Managers",
    "human resources": "Human Resources Specialists",
    "engineer": "Software Developers",
    "tester": "Software Quality Assurance Analysts and Testers",
    "helpdesk": "Computer User Support Specialists",
    "support engineer": "Computer User Support Specialists",
}


def map_role_to_occupation(role, occupation_df):
    """Map an employee's job role to the closest O*NET occupation."""
    role_lower = role.lower().strip()

    # Try exact hint match
    if role_lower in ROLE_TO_OCCUPATION_HINTS:
        target_title = ROLE_TO_OCCUPATION_HINTS[role_lower]
        matches = occupation_df[occupation_df["Title"].str.contains(
            target_title.split(",")[0], case=False, na=False
        )]
        if not matches.empty:
            return matches.iloc[0]["O*NET-SOC Code"], matches.iloc[0]["Title"]

    # Fuzzy search across occupation titles
    best_score = 0
    best_occ = None
    for _, row in occupation_df.iterrows():
        score = fuzzy_match_score(role, row["Title"])
        if score > best_score:
            best_score = score
            best_occ = row

    if best_occ is not None and best_score > 0.3:
        return best_occ["O*NET-SOC Code"], best_occ["Title"]

    return None, None


def infer_employee_skills(role, occupation_df, essential_df, software_df):
    """
    Infer an employee's current skills based on their job role.
    Maps role → O*NET occupation → required skills for that occupation.
    Returns (skills_list, occupation_title, is_inferred).
    """
    soc_code, occ_title = map_role_to_occupation(role, occupation_df)
    if soc_code is None:
        return [], "Unknown", True

    skills = get_combined_skills_for_occupation(essential_df, software_df, soc_code)
    return skills, occ_title, True


def calculate_readiness_score(matched, total_required):
    """Calculate career readiness score as percentage."""
    if total_required == 0:
        return 100.0
    return round((len(matched) / total_required) * 100, 1)
