import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="centered")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan. Pastikan best_model.pkl, scaler.pkl, dan "
        "feature_columns.pkl ada di folder yang sama dengan app.py."
    )
    st.stop()

st.title("🏭 Predictive Maintenance")
st.caption("Prediksi apakah mesin butuh maintenance berdasarkan parameter proses produksi.")

tab_single, tab_batch = st.tabs(["Prediksi Manual", "Prediksi dari CSV"])

# ---------- Tab 1: manual single input ----------
with tab_single:
    st.subheader("Masukkan parameter mesin")

    input_values = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_columns):
        with cols[i % 2]:
            # ganti step/format sesuai kebutuhan tiap fitur kalau perlu
            input_values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0,
                step=0.1,
                format="%.3f",
            )

    if st.button("Prediksi", type="primary"):
        input_df = pd.DataFrame([input_values])[feature_columns]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Mesin diprediksi **BUTUH MAINTENANCE** (probabilitas {proba:.1%})")
        else:
            st.success(f"✅ Mesin diprediksi **AMAN** (probabilitas butuh maintenance {proba:.1%})")

        st.progress(float(proba))

# ---------- Tab 2: batch prediction from CSV ----------
with tab_batch:
    st.subheader("Upload CSV untuk prediksi batch")
    st.caption(f"CSV harus punya kolom: {', '.join(feature_columns)}")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = set(feature_columns) - set(batch_df.columns)

        if missing_cols:
            st.error(f"Kolom berikut tidak ada di CSV: {', '.join(missing_cols)}")
        else:
            X_batch = batch_df[feature_columns]
            X_batch_scaled = scaler.transform(X_batch)

            batch_df["prediction"] = model.predict(X_batch_scaled)
            batch_df["probability_maintenance"] = model.predict_proba(X_batch_scaled)[:, 1]

            st.success(f"Berhasil memprediksi {len(batch_df)} baris.")
            st.dataframe(batch_df, use_container_width=True)

            csv_result = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil (CSV)",
                data=csv_result,
                file_name="prediction_results.csv",
                mime="text/csv",
            ) 

st.divider()
st.caption("COMPFEST 18 AIC — Predictive Maintenance untuk Industri Tekstil")
