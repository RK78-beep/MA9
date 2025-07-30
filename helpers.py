import pandas as pd
import pdfplumber
import io
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def parse_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    # Dummy DataFrame creation
    data = {'Deal Value (in $M)': [1000], 'Previous Deals': [3], 'Industry': ['Tech'],
            'Target Region': ['US'], 'Merger Type': ['Acquisition']}
    return pd.DataFrame(data)

def parse_excel(file):
    return pd.read_excel(file)

def parse_csv(file):
    return pd.read_csv(file)

def recommend_deal(df):
    model = joblib.load("model.pkl")
    required_cols = ['Deal Value (in $M)', 'Previous Deals', 'Industry', 'Target Region', 'Merger Type']

    for col in required_cols:
        if col not in df.columns:
            df[col] = ['Unknown'] * len(df)

    df = df[required_cols]
    preds = model.predict(df)
    df['Predicted Success'] = preds

    commentary = pd.DataFrame({
        'Prediction': preds,
        'GPT_Commentary': ['High risk. Misalignment in valuation or financials could be concern.' if p == 0 else 'Deal looks promising based on input parameters.' for p in preds]
    })

    return df, commentary

def generate_report(df):
    return df.to_csv(index=False).encode('utf-8')

def plot_summary(df):
    fig, ax = plt.subplots()
    df['Predicted Success'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    st.pyplot(fig)
