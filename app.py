# GxP Validator for Jira - Anomaly Detection with Visual Explorer, Validation Report, SHAP Explainability, and Audit Trail
# Author: Pharma-Focused Software Engineer

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="GxP Validator: Anomaly Detection", layout="wide")

st.title("🧠 GxP Validator for Jira - Anomaly Detection")

# Utility: Write audit logs
def write_audit_log(action):
    with open("audit_log.txt", "a") as log:
        log.write(f"[{datetime.now()}] {action}\n")

# Step 1: File upload
uploaded_file = st.file_uploader("📂 Upload CSV File (Jira/GxP Data with numeric columns)", type="csv")

if uploaded_file:
    write_audit_log("File uploaded")
    st.success("✅ File loaded successfully")
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Raw Data Preview")
    st.write(df.head())

    # Step 2: Feature selection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    st.sidebar.title("⚙️ Configuration")
    selected_features = st.sidebar.multiselect("Select features for anomaly detection", numeric_cols, default=numeric_cols)

    if selected_features:
        write_audit_log(f"Selected features: {selected_features}")

        # Step 3: Train the model
        X = df[selected_features]
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X)
        df['anomaly_score'] = model.decision_function(X)
        df['is_anomaly'] = model.predict(X)
        write_audit_log("Anomaly detection model trained")

        # Step 4: Visual Explorer
        st.subheader("📊 Visual Explorer")
        if 'Temp_C' in df.columns and 'pH' in df.columns:
            fig = px.scatter(
                df, x="Temp_C", y="pH",
                color=df['is_anomaly'].map({1: "Normal", -1: "Anomaly"}),
                symbol=df.get("InjectedAnomaly", False).map(lambda x: "Injected" if x else "Regular"),
                title="Temperature vs pH - Anomaly Detection"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Step 5: Show anomalies
        anomalies = df[df['is_anomaly'] == -1]
        st.subheader(f"🚨 Detected Anomalies: {len(anomalies)}")
        st.dataframe(anomalies)
        write_audit_log(f"Detected {len(anomalies)} anomalies")

        # Step 6: Validation Report
        if "InjectedAnomaly" in df.columns:
            true_anomalies = df[df["InjectedAnomaly"] == True]
            true_detected = true_anomalies[true_anomalies["is_anomaly"] == -1]
            recall = len(true_detected) / len(true_anomalies) * 100 if len(true_anomalies) > 0 else 0
            st.subheader("📈 Validation Report")
            st.markdown(f"**Injected Anomalies:** {len(true_anomalies)}")
            st.markdown(f"**Correctly Detected:** {len(true_detected)}")
            st.markdown(f"**Detection Rate:** {recall:.2f}%")
            write_audit_log("Validation report generated")

        # Step 7: SHAP Explainability
        st.subheader("🔍 Explainable AI - SHAP Values")
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Feature impact for a sample of 100 rows:")
            shap.summary_plot(shap_values[:100], X.iloc[:100], plot_type="bar")
            st.pyplot(bbox_inches='tight')
            write_audit_log("SHAP explainability chart generated")
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
            write_audit_log(f"SHAP explanation failed: {e}")

        # Step 8: Save the model
        if st.button("💾 Save Model"):
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/gxp_anomaly_model.pkl")
            st.success("Model saved successfully.")
            write_audit_log("Model saved")

        # Step 9: Download
        if st.download_button("📤 Download Annotated CSV", df.to_csv(index=False), file_name="annotated_output.csv"):
            write_audit_log("Annotated CSV downloaded")

    else:
        st.warning("Please select at least one numeric column to proceed.")
        write_audit_log("No features selected")

else:
    st.info("Please upload a valid CSV file to begin.")
