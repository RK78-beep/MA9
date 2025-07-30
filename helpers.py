
import pandas as pd
import io
import pdfplumber
import openpyxl
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

model = joblib.load("model.pkl")

def parse_csv(file):
    return pd.read_csv(file)

def parse_excel(file):
    return pd.read_excel(file)

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    # Dummy data from PDF parsing
    return pd.DataFrame({"Deal Value (in $M)": [1000], "Previous Deals": [5], "Industry": ["Tech"], "Target Region": ["US"], "Merger Type": ["Acquisition"]})

def recommend_deal(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    score = model.predict_proba(df_encoded)[:, 1].mean()
    alt = "Consider targeting a different region or changing deal type."
    recommendation = "This deal has potential synergies and fits industry patterns." if score >= 0.5 else "This deal poses risks due to mismatch in strategy or financial history."
    return score, alt, recommendation

def generate_charts(df):
    st.bar_chart(df.select_dtypes(include='number'))

def explain_with_shap(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    explainer = shap.Explainer(model)
    shap_values = explainer(df_encoded)
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")
