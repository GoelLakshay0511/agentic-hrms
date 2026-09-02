"""
Agentic HRMS — Learning Recommendations Engine
Course catalog generation and skill-gap-to-course mapping.
"""
import pandas as pd

# ─── Internal Course Catalog ─────────────────────────────────────────────────

COURSE_CATALOG = [
    # Data & Analytics
    {"course_name": "Python for Data Analysis", "skill": "Python", "category": "Data & Analytics", "difficulty": "Beginner", "duration_hours": 20, "description": "Master Python fundamentals for data manipulation, analysis, and visualization using pandas, NumPy, and matplotlib."},
    {"course_name": "Advanced SQL Masterclass", "skill": "SQL", "category": "Data & Analytics", "difficulty": "Intermediate", "duration_hours": 15, "description": "Complex queries, window functions, CTEs, optimization, and database design patterns."},
    {"course_name": "Statistical Analysis & Inference", "skill": "Statistics", "category": "Data & Analytics", "difficulty": "Intermediate", "duration_hours": 25, "description": "Hypothesis testing, regression analysis, probability distributions, and experimental design."},
    {"course_name": "Machine Learning Fundamentals", "skill": "Machine Learning", "category": "Data & Analytics", "difficulty": "Intermediate", "duration_hours": 40, "description": "Supervised and unsupervised learning algorithms, model evaluation, and feature engineering."},
    {"course_name": "Deep Learning with PyTorch", "skill": "Deep Learning", "category": "Data & Analytics", "difficulty": "Advanced", "duration_hours": 35, "description": "Neural network architectures, CNNs, RNNs, transformers, and practical deep learning projects."},
    {"course_name": "Data Visualization & Storytelling", "skill": "Data Visualization", "category": "Data & Analytics", "difficulty": "Beginner", "duration_hours": 12, "description": "Effective chart design, dashboard creation, and communicating insights with Tableau and Power BI."},
    # Software Engineering
    {"course_name": "Full-Stack Web Development", "skill": "Web Development", "category": "Software Engineering", "difficulty": "Intermediate", "duration_hours": 60, "description": "HTML, CSS, JavaScript, React, Node.js, and RESTful API development."},
    {"course_name": "Cloud Computing with AWS", "skill": "Cloud Computing", "category": "Software Engineering", "difficulty": "Intermediate", "duration_hours": 30, "description": "AWS services, cloud architecture, deployment, scaling, and cost optimization."},
    {"course_name": "Docker & Container Orchestration", "skill": "Docker", "category": "Software Engineering", "difficulty": "Intermediate", "duration_hours": 18, "description": "Container fundamentals, Docker Compose, Kubernetes basics, and CI/CD pipelines."},
    {"course_name": "Cybersecurity Essentials", "skill": "Cybersecurity", "category": "Software Engineering", "difficulty": "Beginner", "duration_hours": 22, "description": "Network security, encryption, threat modeling, vulnerability assessment, and security best practices."},
    {"course_name": "Software Quality & Testing", "skill": "Software Testing", "category": "Software Engineering", "difficulty": "Intermediate", "duration_hours": 16, "description": "Unit testing, integration testing, test automation, and quality assurance methodologies."},
    # Business & Management
    {"course_name": "Digital Marketing Strategy", "skill": "Digital Marketing", "category": "Business & Management", "difficulty": "Beginner", "duration_hours": 15, "description": "SEO, SEM, social media marketing, content strategy, and analytics."},
    {"course_name": "Financial Analysis & Modeling", "skill": "Financial Analysis", "category": "Business & Management", "difficulty": "Intermediate", "duration_hours": 25, "description": "Financial statement analysis, valuation, DCF modeling, and investment analysis."},
    {"course_name": "Project Management Professional", "skill": "Project Management", "category": "Business & Management", "difficulty": "Intermediate", "duration_hours": 35, "description": "Agile, Scrum, Waterfall, risk management, stakeholder communication, and PMP preparation."},
    {"course_name": "Strategic Leadership", "skill": "Leadership", "category": "Business & Management", "difficulty": "Advanced", "duration_hours": 20, "description": "Executive leadership, change management, organizational development, and strategic planning."},
    {"course_name": "Business Communication Excellence", "skill": "Communication", "category": "Business & Management", "difficulty": "Beginner", "duration_hours": 10, "description": "Professional writing, presentation skills, negotiation, and cross-cultural communication."},
    # Core Skills
    {"course_name": "Critical Thinking & Problem Solving", "skill": "Critical Thinking", "category": "Core Skills", "difficulty": "Beginner", "duration_hours": 12, "description": "Analytical reasoning, logical frameworks, decision matrices, and structured problem solving."},
    {"course_name": "Active Listening & Communication", "skill": "Active Listening", "category": "Core Skills", "difficulty": "Beginner", "duration_hours": 8, "description": "Empathetic listening techniques, feedback skills, and effective interpersonal communication."},
    {"course_name": "Mathematics for Professionals", "skill": "Mathematics", "category": "Core Skills", "difficulty": "Intermediate", "duration_hours": 20, "description": "Applied math concepts including algebra, calculus fundamentals, and quantitative reasoning."},
    {"course_name": "Technical Writing Masterclass", "skill": "Writing", "category": "Core Skills", "difficulty": "Intermediate", "duration_hours": 14, "description": "Documentation, technical reports, proposals, SOPs, and professional writing standards."},
    {"course_name": "Scientific Method & Research", "skill": "Science", "category": "Core Skills", "difficulty": "Intermediate", "duration_hours": 18, "description": "Research methodology, experimental design, data collection, and scientific communication."},
    {"course_name": "Learning Strategies & Self-Development", "skill": "Learning Strategies", "category": "Core Skills", "difficulty": "Beginner", "duration_hours": 8, "description": "Metacognition, accelerated learning, skill acquisition frameworks, and continuous improvement."},
    {"course_name": "Reading Comprehension for Professionals", "skill": "Reading Comprehension", "category": "Core Skills", "difficulty": "Beginner", "duration_hours": 6, "description": "Speed reading, analytical reading, summarization, and extracting key insights from complex documents."},
    {"course_name": "Presentation & Public Speaking", "skill": "Speaking", "category": "Core Skills", "difficulty": "Beginner", "duration_hours": 10, "description": "Public speaking, storytelling, slide design, and confident presentation delivery."},
    {"course_name": "Performance Monitoring & Evaluation", "skill": "Monitoring", "category": "Core Skills", "difficulty": "Intermediate", "duration_hours": 12, "description": "KPI design, performance tracking systems, evaluation frameworks, and continuous monitoring."},
    # Technology Specific
    {"course_name": "Microsoft Excel Advanced Analytics", "skill": "Microsoft Excel", "category": "Technology", "difficulty": "Intermediate", "duration_hours": 15, "description": "Advanced formulas, pivot tables, Power Query, VBA macros, and data modeling."},
    {"course_name": "Salesforce CRM Administration", "skill": "Salesforce", "category": "Technology", "difficulty": "Intermediate", "duration_hours": 25, "description": "Salesforce setup, customization, workflows, reports, and admin certification preparation."},
    {"course_name": "SAP ERP Fundamentals", "skill": "SAP", "category": "Technology", "difficulty": "Intermediate", "duration_hours": 30, "description": "SAP modules overview, navigation, basic configuration, and business process integration."},
    {"course_name": "Tableau Data Visualization", "skill": "Tableau", "category": "Technology", "difficulty": "Beginner", "duration_hours": 16, "description": "Tableau Desktop, calculated fields, dashboard design, and interactive visual analytics."},
    {"course_name": "R Programming for Analytics", "skill": "R Programming", "category": "Technology", "difficulty": "Intermediate", "duration_hours": 22, "description": "R fundamentals, tidyverse, ggplot2, statistical modeling, and R Markdown reporting."},
]


def get_course_catalog():
    """Return the course catalog as a DataFrame."""
    return pd.DataFrame(COURSE_CATALOG)


def recommend_courses_for_gaps(skill_gaps, n_per_skill=1):
    """
    Recommend courses for identified skill gaps.
    Returns list of recommendations with reasoning.
    """
    catalog = get_course_catalog()
    recommendations = []

    for gap_skill in skill_gaps:
        gap_lower = gap_skill.lower().strip()

        # Score each course by relevance to the gap skill
        best_score = 0
        best_course = None

        for _, course in catalog.iterrows():
            # Check skill name match
            course_skill = course["skill"].lower()
            course_name = course["course_name"].lower()
            course_desc = course["description"].lower()

            score = 0
            if gap_lower == course_skill:
                score = 1.0
            elif gap_lower in course_skill or course_skill in gap_lower:
                score = 0.8
            elif gap_lower in course_name:
                score = 0.6
            elif gap_lower in course_desc:
                score = 0.4
            elif any(w in course_skill for w in gap_lower.split()):
                score = 0.3

            if score > best_score:
                best_score = score
                best_course = course

        if best_course is not None and best_score > 0.2:
            recommendations.append({
                "skill_gap": gap_skill,
                "course_name": best_course["course_name"],
                "skill": best_course["skill"],
                "category": best_course["category"],
                "difficulty": best_course["difficulty"],
                "duration_hours": best_course["duration_hours"],
                "description": best_course["description"],
                "relevance_score": round(best_score, 2),
                "reason": f"Addresses the '{gap_skill}' skill gap through structured learning in {best_course['skill']}.",
            })
        else:
            # Generic recommendation
            recommendations.append({
                "skill_gap": gap_skill,
                "course_name": f"Self-Directed Study: {gap_skill}",
                "skill": gap_skill,
                "category": "Self-Directed",
                "difficulty": "Varies",
                "duration_hours": 15,
                "description": f"Self-paced learning program focused on developing proficiency in {gap_skill}.",
                "relevance_score": 0.1,
                "reason": f"No specialized course found; recommending self-directed study for '{gap_skill}'.",
            })

    # Sort by relevance
    recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
    return recommendations


def generate_learning_path(recommendations):
    """
    Generate a prioritized learning path from recommendations.
    Orders by: difficulty (beginner first), then relevance.
    """
    difficulty_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2, "Varies": 1}

    path = sorted(
        recommendations,
        key=lambda x: (difficulty_order.get(x["difficulty"], 1), -x["relevance_score"])
    )

    for i, item in enumerate(path):
        item["priority"] = i + 1
        item["phase"] = "Foundation" if item["difficulty"] == "Beginner" else (
            "Core Development" if item["difficulty"] == "Intermediate" else "Advanced Specialization"
        )

    return path


def get_certification_suggestions(skill_gaps):
    """Suggest relevant certifications based on skill gaps."""
    cert_map = {
        "python": "Python Institute PCEP / PCAP Certification",
        "sql": "Oracle SQL Certified Associate",
        "machine learning": "AWS Machine Learning Specialty",
        "deep learning": "NVIDIA Deep Learning Institute Certification",
        "cloud computing": "AWS Solutions Architect Associate",
        "cybersecurity": "CompTIA Security+ Certification",
        "project management": "PMP (Project Management Professional)",
        "data visualization": "Tableau Desktop Specialist",
        "financial analysis": "CFA (Chartered Financial Analyst) Level 1",
        "docker": "Docker Certified Associate",
        "salesforce": "Salesforce Administrator Certification",
        "sap": "SAP Certified Application Associate",
        "digital marketing": "Google Digital Marketing Certificate",
        "leadership": "CCL (Center for Creative Leadership) Certificate",
    }

    suggestions = []
    for gap in skill_gaps:
        gap_lower = gap.lower()
        for key, cert in cert_map.items():
            if key in gap_lower or gap_lower in key:
                suggestions.append({"skill": gap, "certification": cert})
                break

    return suggestions
