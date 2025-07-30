import pandas as pd

def parse_pdf(file):
    # Dummy parser for PDF files
    return pd.DataFrame()

def parse_excel(file):
    return pd.read_excel(file)

def parse_csv(file):
    return pd.read_csv(file)

def generate_gpt_commentary(prediction):
    if prediction == 1:
        return "Low risk. Deal seems financially and strategically sound."
    else:
        return "High risk. Misalignment in valuation or financials could be concern."

def recommend_deal(df):
    if df["Prediction"].mean() > 0.5:
        return "Proceed with the M&A Deal"
    else:
        return "Do Not Proceed with the M&A Deal"
