
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

model = joblib.load("model.pkl")

def process_uploaded_file(file):
    ext = file.name.split(".")[-1]
    if ext in ["xlsx", "xls"]:
        df = pd.read_excel(file)
    elif ext == "csv":
        df = pd.read_csv(file)
    elif ext == "pdf":
        # Placeholder: Replace with real PDF parser logic
        raise ValueError("PDF parsing is not yet implemented.")
    else:
        raise ValueError("Unsupported file type.")
    return df

def generate_predictions(df):
    required = ["Deal Value (in $M)", "Previous Deals", "Industry", "Target Region", "Merger Type"]
    if not all(col in df.columns for col in required):
        missing = list(set(required) - set(df.columns))
        raise ValueError(f"Missing columns: {missing}")
    X = df[required]
    preds = model.predict(X)
    return preds

def generate_recommendations(df):
    df["GPT_Commentary"] = df["Predicted Success"].apply(
        lambda x: "✅ Good fit. Synergies possible." if x else "⚠️ High risk. Misalignment likely."
    )
    return df[["Predicted Success", "GPT_Commentary"]]

def generate_report(df, commentary):
    # Placeholder: Generate a simple PDF report
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="M&A Deal Analyzer Report", ln=True)
    pdf.cell(200, 10, txt=f"Total Deals Evaluated: {len(df)}", ln=True)
    pdf.cell(200, 10, txt="Summary:", ln=True)
    for i in range(min(10, len(commentary))):
        pdf.cell(200, 10, txt=f"{commentary.iloc[i]['GPT_Commentary']}", ln=True)
    out = BytesIO()
    pdf.output(out)
    return out.getvalue()
