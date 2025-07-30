
import streamlit as st
import pandas as pd
import base64
from helpers import parse_pdf, parse_excel, parse_csv, recommend_deal, generate_commentary, make_prediction, show_shap_values, smart_column_parser

# App title
st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")
st.title("📊 M&A Deal Analyzer+")
st.markdown("Upload your M&A deal data (CSV, Excel, or PDF) and get a smart prediction with GPT-style insights.")

# File uploader
uploaded_file = st.file_uploader("Upload M&A Data File", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = parse_excel(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        df = parse_csv(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    if df is not None:
        st.subheader("🧾 Uploaded & Processed Data")
        st.dataframe(df)

        # Smart column parser
        df_parsed = smart_column_parser(df)

        # Prediction
        st.subheader("📈 Deal Success Prediction")
        try:
            preds, probas = make_prediction(df_parsed)
            df["Prediction"] = preds
            df["Probability"] = probas

            st.bar_chart(df["Prediction"].value_counts())

            # GPT-style commentary
            st.subheader("🧠 AI Commentary")
            df["GPT_Commentary"] = df.apply(generate_commentary, axis=1)
            st.dataframe(df[["Prediction", "GPT_Commentary"]])

            # SHAP values visualization
            st.subheader("📌 Explainability (SHAP Values)")
            show_shap_values(df_parsed)

            # Recommendation summary
            st.subheader("✅ Final Recommendation")
            recommendation = recommend_deal(preds, probas)
            st.success(recommendation)

            # Downloadable report
            st.download_button("Download Analysis Report", df.to_csv(index=False), file_name="ma_analysis_report.csv")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Could not parse the uploaded file.")
