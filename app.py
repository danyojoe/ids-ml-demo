import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NSL-KDD Intrusion Detection Demo", layout="wide")

st.title("Intrusion Detection (NSL-KDD) — ML Demo")
st.write("Upload a CSV of network records and the model will predict **Normal** or **Intrusion** + probability.")

@st.cache_resource
def load_model():
    return joblib.load("rf_ids_model.pkl")

model = load_model()

st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("Tip: CSV must contain the same feature columns as the NSL-KDD dataset (without label/difficulty).")

if uploaded is None:
    st.warning("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(10), use_container_width=True)

# Predict
try:
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]  # probability of class 1 (attack)

    out = df.copy()
    out["prediction"] = ["INTRUSION" if p == 1 else "NORMAL" for p in preds]
    out["attack_probability"] = probs.round(4)

    st.subheader("Predictions")
    st.dataframe(out.head(20), use_container_width=True)

    # Summary stats
    st.subheader("Summary")
    intrusions = (preds == 1).sum()
    normals = (preds == 0).sum()
    st.write(f"Normal: **{normals}** | Intrusion: **{intrusions}** | Total: **{len(preds)}**")

    # Download results
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name="ids_predictions.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error("Prediction failed. Most likely: your CSV columns don't match what the model expects.")
    st.code(str(e))
    st.stop()
