# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:51:25 2026

@author: pc
"""

import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os
import base64
from pathlib import Path
from pipeline import run_contract_analysis  # your updated backend

# -------------------- Background --------------------
def set_bg():
    image_path = Path("C:/Users/pc/Documents/aiphoto.jpg")
    if not image_path.exists():
        st.error("Background image not found. Make sure ai_bg.png is in the same folder as app.py")
        return

    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(0, 0, 0, 0.75);
            padding: 2rem;
            border-radius: 16px;
        }}
        h1, h2, h3, h4 {{
            color: #00f5ff;
            text-shadow: 0 0 12px rgba(0, 245, 255, 0.9);
        }}
        p, span, div {{
            color: #e0e0e0;
        }}
        button {{
            background: linear-gradient(90deg, #00f5ff, #0066ff) !important;
            color: black !important;
            border-radius: 8px !important;
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# -------------------- Risk Badge --------------------
def risk_badge(risk_level: str) -> str:
    color_map = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}
    color = color_map.get(risk_level.upper(), "#6c757d")
    return f'<span style="background-color:{color}; color:white; padding:3px 8px; border-radius:5px;">{risk_level}</span>'

# -------------------- PDF Generation --------------------
def generate_pdf(report_data):
    styles = getSampleStyleSheet()
    elements = []

    title = styles["Title"]
    normal = styles["Normal"]

    elements.append(Paragraph("Contract Risk Analysis Report", title))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%m-%Y')}", normal))
    elements.append(Spacer(1, 12))

    # Executive Summary
    overview = report_data["overview"]
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Overall Risk:</b> {overview['overall_risk']}", normal))
    elements.append(Paragraph(overview["summary"], normal))
    elements.append(Spacer(1, 12))

    def section(title, items):
        elements.append(Paragraph(title, styles["Heading2"]))
        for item in items:
            for k, v in item.items():
                elements.append(Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", normal))
            elements.append(Spacer(1, 12))

    section("Finance Findings", report_data["finance"])
    section("Legal Findings", report_data["legal"])
    section("Operations Findings", report_data["operations"])
    section("Compliance Findings", report_data["compliance"])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    doc.build(elements)
    return tmp.name

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Contract Intelligence Dashboard", layout="wide")
st.title("📄 AI TOOL TO ANALYZE CONTRACT DOCUMENTS")
st.caption("Multi-Agent Contract Risk Analysis System")

uploaded_file = st.file_uploader("Upload Contract (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Run Analysis"):
        with st.spinner("Running Finance, Legal, Operations and Compliance Agents..."):
            report = run_contract_analysis(temp_path)
            st.session_state["report"] = report

# -------------------- Output Tabs --------------------
if "report" in st.session_state:
    report = st.session_state["report"]

    tabs = st.tabs(["Overview", "Finance", "Legal", "Operations", "Compliance", "Final Report"])

    # ---------- Overview ----------
    with tabs[0]:
        st.markdown(f"**Overall Risk Level:** {risk_badge(report['overview']['overall_risk'])}", unsafe_allow_html=True)
        st.write(report["overview"]["summary"])

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"**Finance Risk:** {risk_badge(report['overview']['finance_risk'])}", unsafe_allow_html=True)
        col2.markdown(f"**Legal Risk:** {risk_badge(report['overview']['legal_risk'])}", unsafe_allow_html=True)
        col3.markdown(f"**Operations Risk:** {risk_badge(report['overview']['operations_risk'])}", unsafe_allow_html=True)
        col4.markdown(f"**Compliance Risk:** {risk_badge(report['overview']['compliance_risk'])}", unsafe_allow_html=True)

    # ---------- Finance ----------
    with tabs[1]:
        for f in report["finance"]:
            st.markdown(f"**Risk Level:** {risk_badge(f['risk_level'])}", unsafe_allow_html=True)
            st.markdown(f"**Impact:** {f.get('impact', 'N/A')}")
            st.markdown(f"**Recommendation:** {f.get('recommendation', 'N/A')}")
            st.divider()

    # ---------- Legal ----------
    with tabs[2]:
        for l in report["legal"]:
            st.markdown(f"**Risk Level:** {risk_badge(l['risk_level'])}", unsafe_allow_html=True)
            st.markdown(f"**Issue:** {l.get('issue', 'N/A')}")
            st.markdown(f"**Explanation:** {l.get('explanation', 'N/A')}")
            st.markdown(f"**Recommendation:** {l.get('recommendation', 'N/A')}")
            st.divider()

    # ---------- Operations ----------
    with tabs[3]:
        for o in report["operations"]:
            st.markdown(f"**Risk Level:** {risk_badge(o['risk_level'])}", unsafe_allow_html=True)
            st.markdown(f"**Type:** {o.get('type', 'N/A')}")
            st.markdown(f"**Impact:** {o.get('impact', 'N/A')}")
            st.markdown(f"**Action Needed:** {o.get('action', 'N/A')}")
            st.divider()

    # ---------- Compliance ----------
    with tabs[4]:
        for c in report["compliance"]:
            st.markdown(f"**Risk Level:** {risk_badge(c['risk_level'])}", unsafe_allow_html=True)
            st.markdown(f"**Area:** {c.get('area', 'N/A')}")
            st.markdown(f"**Violation:** {c.get('violation', 'N/A')}")
            st.markdown(f"**Required Action:** {c.get('required_action', 'N/A')}")
            st.divider()

    # ---------- Final Report + PDF ----------
    with tabs[5]:
        st.write("This content will be exported as a professional PDF report.")
        pdf_path = generate_pdf(report)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Final Report as PDF",
                data=f,
                file_name="contract_risk_report.pdf",
                mime="application/pdf"
            )

        st.divider()
        st.subheader("Feedback")
        rating = st.slider("How useful was this analysis?", 1, 5, 3)
        feedback = st.text_area("What was unclear or missing?")
        if st.button("Submit Feedback"):
            st.success("Feedback recorded.")
