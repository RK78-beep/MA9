import streamlit as st
from helpers import parse_pdf, parse_excel, parse_csv, recommend_deal
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")

st.title("🤖 M&A Deal Analyzer+")

uploaded_file = st.file_uploader("Upload your M&A deal file (PDF, Excel, or CSV)", type=["pdf", "xlsx", "xls", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = parse_excel(uploaded_file)
    else:
        df = parse_csv(uploaded_file)

    st.subheader("📊 Uploaded & Processed Data")
    st.dataframe(df)

    model = joblib.load("model.pkl")
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)[:, 1]

    df["Success Probability"] = prediction_proba
    df["Prediction"] = prediction

    st.subheader("✅ Prediction Results")
    st.dataframe(df[["Success Probability", "Prediction"]])

    st.subheader("📈 SHAP Explainability")
    explainer = shap.Explainer(model)
    shap_values = explainer(df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(bbox_inches='tight')

    st.subheader("🧠 GPT-style Recommendation")
    for i, row in df.iterrows():
        st.markdown(f"**Deal {i+1}:** {recommend_deal(row)}")
