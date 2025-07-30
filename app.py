
import streamlit as st
import pandas as pd
from utils.helpers import handle_file_upload, run_model_analysis

st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")

st.title("🤝 M&A Deal Analyzer+ with AI Commentary")

uploaded_file = st.file_uploader("Upload M&A Financial Document (CSV, Excel, or PDF)", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    try:
        df = handle_file_upload(uploaded_file)
        st.subheader("📄 Uploaded & Processed Data")
        st.dataframe(df)

        st.subheader("🔍 Analysis & Prediction")
        run_model_analysis(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
