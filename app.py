
import streamlit as st
import pandas as pd
from helpers import process_uploaded_file, generate_predictions, generate_recommendations, generate_report

st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")

st.title("📊 M&A Deal Analyzer+")

uploaded_file = st.file_uploader("Upload Financial Document (CSV, Excel, or PDF)", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    try:
        df = process_uploaded_file(uploaded_file)
        st.subheader("📁 Uploaded & Processed Data")
        st.dataframe(df)

        predictions = generate_predictions(df)
        df["Predicted Success"] = predictions

        st.subheader("✅ Predictions")
        st.dataframe(df)

        st.subheader("📉 SHAP Analysis (Explainability)")
        st.image("shap_summary_plot.png")

        st.subheader("📋 GPT-style Commentary")
        commentary = generate_recommendations(df)
        st.dataframe(commentary)

        st.download_button("📥 Download Report", generate_report(df, commentary), file_name="ma_analysis_report.pdf")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
