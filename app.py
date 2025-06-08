# GxP Validator for Jira - Anomaly Detection Proof-of-Concept
# Author: Pharma-Focused Software Engineer (20+ years experience)
# Language: Python 3.x
# Frameworks: Streamlit (UI), scikit-learn (ML), pandas (data), joblib (model storage)

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

st.set_page_config(page_title="GxP Validator: Anomaly Detection", layout="wide")

st.title("🧠 GxP Validator for Jira - Anomaly Detection")

# Step 1: File upload
uploaded_file = st.file_uploader("📂 Upload CSV File (Jira/GxP Data)", type="csv")

if uploaded_file:
    st.success("✅ File loaded successfully")
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Raw Data Preview")
    st.write(df.head())

    # Step 2: Feature selection (auto-detect numeric columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    st.sidebar.title("⚙️ Configuration")
    selected_features = st.sidebar.multiselect("Select features for anomaly detection", numeric_cols, default=numeric_cols)

    if selected_features:
        # Step 3: Train the model
        X = df[selected_features]
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X)
        df['anomaly_score'] = model.decision_function(X)
        df['is_anomaly'] = model.predict(X)

        # Step 4: Display anomalies
        anomalies = df[df['is_anomaly'] == -1]
        st.subheader(f"🚨 Detected Anomalies: {len(anomalies)}")
        st.dataframe(anomalies)

        # Step 5: Save the model
        if st.button("💾 Save Model"):
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/gxp_anomaly_model.pkl")
            st.success("Model saved successfully.")

        # Step 6: Export annotated data
        if st.download_button("📤 Download Results as CSV", df.to_csv(index=False), file_name="annotated_output.csv"):
            st.toast("Results downloaded")

    else:
        st.warning("Please select at least one numeric column to proceed.")

else:
    st.info("Please upload a valid CSV file to begin.")
