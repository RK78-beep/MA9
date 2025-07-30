import streamlit as st
from utils.helpers import parse_pdf, parse_excel, parse_csv, recommend_deal, generate_report, plot_summary
import pandas as pd

st.title("📊 M&A Deal Analyzer+ with GPT Insights")

uploaded_file = st.file_uploader("Upload your M&A file (CSV, Excel, PDF)", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'pdf':
        df = parse_pdf(uploaded_file)
    elif file_type in ['xlsx', 'xls']:
        df = parse_excel(uploaded_file)
    elif file_type == 'csv':
        df = parse_csv(uploaded_file)
    else:
        st.error("Unsupported file format.")

    if df is not None:
        st.subheader("📁 Uploaded & Processed Data")
        st.dataframe(df)

        # Generate recommendation and report
        df, commentary = recommend_deal(df)
        st.subheader("🤖 GPT-Style Commentary")
        st.dataframe(commentary)

        st.subheader("📈 Prediction Distribution")
        plot_summary(df)

        st.download_button("Download Report", generate_report(df), "report.csv", "text/csv")
