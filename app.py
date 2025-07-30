
import streamlit as st
import pandas as pd
from helpers import parse_csv, parse_excel, parse_pdf, recommend_deal, generate_charts, explain_with_shap

st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")
st.title("🤖 M&A Deal Analyzer with AI Insights")

uploaded_file = st.file_uploader("Upload your deal data file (CSV, Excel, or PDF)", type=["csv", "xlsx", "xls", "pdf"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = parse_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = parse_excel(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    if df is not None:
        st.subheader("📊 Uploaded & Processed Data")
        st.dataframe(df)

        try:
            score, alt, recommendation = recommend_deal(df)
            st.success(f"✅ Success Probability: {score:.2f}")
            st.markdown(f"**💡 GPT-Style Recommendation:** {recommendation}")
            if score < 0.5:
                st.warning(f"🚧 Suggested Alternatives: {alt}")

            st.subheader("📈 Visual Insights")
            generate_charts(df)

            st.subheader("🧠 SHAP Explainability")
            explain_with_shap(df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("No valid data extracted.")
