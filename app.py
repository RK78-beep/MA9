import streamlit as st
from helpers import parse_file, predict_and_explain, generate_report

st.set_page_config(layout="wide")
st.title("🤖 M&A Deal Analyzer+")

uploaded_file = st.file_uploader("Upload a CSV, Excel or PDF file for the target company", type=["csv", "xlsx", "xls", "pdf"])
if uploaded_file:
    try:
        df = parse_file(uploaded_file)
        st.subheader("Uploaded & Processed Data")
        st.dataframe(df)
        pred, shap_fig, commentary, decision, report_df = predict_and_explain(df)
        st.subheader("Prediction Score (0 = Fail, 1 = Success)")
        st.metric("Success Probability", f"{pred:.2f}")
        st.subheader("Recommendation")
        st.write(commentary)
        st.success(f"💡 Final Suggestion: {decision}")
        st.subheader("Explainability (SHAP)")
        st.pyplot(shap_fig)
        st.download_button("Download Report", generate_report(report_df), "report.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")