import pandas as pd
import random

def parse_csv(file):
    return pd.read_csv(file)

def parse_excel(file):
    return pd.read_excel(file)

def parse_pdf(file):
    return pd.DataFrame({
        "Deal Value (in $M)": [1000],
        "Previous Deals": [3],
        "Industry": ["Tech"],
        "Target Region": ["Asia"],
        "Merger Type": ["Acquisition"]
    })

def recommend_deal(row):
    prob = row["Success Probability"]
    if prob > 0.75:
        return f"High probability of success. Proceed with confidence. Industry: {row['Industry']}."
    elif prob > 0.4:
        return f"Moderate risk. Consider additional due diligence for region: {row['Target Region']}."
    else:
        return f"High risk. Explore alternative targets or renegotiate deal terms."
