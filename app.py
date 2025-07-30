import streamlit as st
import pandas as pd
import pickle
from utils.helpers import parse_pdf, parse_excel, parse_csv, recommend_deal, generate_gpt_commentary

st.title("📊 M&A Deal Analyzer+")
uploaded_file = st.file_uploader("Upload a CSV, Excel, or PDF file", type=["csv", "xlsx", "xls", "pdf"])

if uploaded_file:
    file_name = uploaded_file.name
    if file_name.endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = parse_excel(uploaded_file)
    elif file_name.endswith(".csv"):
        df = parse_csv(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    if df is not None:
        st.subheader("✅ Uploaded & Processed Data")
        st.dataframe(df)

        try:
            model = pickle.load(open("model.pkl", "rb"))
            preds = model.predict(df)
            df["Prediction"] = preds
            df["GPT_Commentary"] = df["Prediction"].apply(generate_gpt_commentary)
            st.subheader("📈 Predictions & AI Commentary")
            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")

            recommendation = recommend_deal(df)
            st.success(f"🤖 Final Recommendation: {recommendation}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
