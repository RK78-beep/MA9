
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from PyPDF2 import PdfReader
import io

def handle_file_upload(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        text = ''
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        lines = text.split('\n')
        rows = [line.split() for line in lines if line.strip()]
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        raise ValueError("Unsupported file format.")
    return df

def run_model_analysis(df):
    model = joblib.load("models/model.pkl")
    required_columns = ['Deal Value (in $M)', 'Previous Deals', 'Industry', 'Target Region', 'Merger Type']

    # Smart column matching
    mapping = {}
    for col in required_columns:
        for uploaded_col in df.columns:
            if col.lower().replace(" ", "") in uploaded_col.lower().replace(" ", ""):
                mapping[col] = uploaded_col
                break

    if len(mapping) < len(required_columns):
        st.warning("Some columns required for prediction are missing or mismatched.")
        return

    df_renamed = df.rename(columns={v: k for k, v in mapping.items()})
    X = df_renamed[required_columns]

    preds = model.predict_proba(X)[:, 1]
    df_renamed['Prediction'] = preds

    def gpt_style_comment(p):
        if p < 0.3:
            return "⚠️ High risk. Misalignment in valuation or industry strategy."
        elif p < 0.6:
            return "⚖️ Moderate potential. Needs deeper due diligence or alternative structuring."
        else:
            return "✅ Promising deal. Strategic and financial synergy likely."

    df_renamed["AI_Commentary"] = df_renamed["Prediction"].apply(gpt_style_comment)

    st.bar_chart(df_renamed["Prediction"])
    st.subheader("🧠 GPT-Style AI Commentary")
    st.dataframe(df_renamed[["Prediction", "AI_Commentary"]])

    # SHAP Analysis
    try:
        explainer = shap.Explainer(model.named_steps["classifier"])
        shap_values = explainer(model.named_steps["preprocessor"].transform(X))
        st.subheader("📊 SHAP Feature Importance")
        shap.plots.bar(shap_values, show=False)
        st.pyplot(bbox_inches='tight')
    except:
        st.warning("SHAP explanation failed. Possibly due to pipeline incompatibility.")
