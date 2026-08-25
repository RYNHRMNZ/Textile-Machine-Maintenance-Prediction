import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_autorefresh import st_autorefresh

from sensor_simulator import TextileMachineSimulator
import os

model_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="centered")

# ---------- Load model artifacts ----------
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load("best_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     feature_columns = joblib.load("feature_columns.pkl")
#     return model, scaler, feature_columns

# try:
#     model, scaler, feature_columns = load_artifacts()
# except FileNotFoundError:
#     st.error(
#         "File model tidak ditemukan. Pastikan best_model.pkl, scaler.pkl, dan "
#         "feature_columns.pkl ada di folder yang sama dengan streamlit_app.py."
#     )
#     st.stop()

st.title("🏭 Predictive Maintenance")
st.caption("Prediksi apakah mesin butuh maintenance berdasarkan parameter proses produksi.")

st.subheader("Live sensor feed & prediksi real-time")

col_a, col_b = st.columns([1, 3])
with col_a:
    interval = st.slider("Refresh interval (detik)", 1, 10, 2, key="rt_interval")
    running = st.toggle("Stream aktif", value=True, key="rt_running")
    threshold = st.slider(
        "Threshold alert (probabilitas)", 0.0, 1.0, 0.5, 0.05, key="rt_threshold"
    )

if "rt_sim" not in st.session_state:
    st.session_state.rt_sim = TextileMachineSimulator()
    st.session_state.rt_data = pd.DataFrame()

if running:
    st_autorefresh(interval=interval * 1000, key="rt_refresh_timer")
    reading = st.session_state.rt_sim.next_reading()

    # Only feed the model the columns it was trained on.
    missing_cols = [c for c in feature_columns if c not in reading]
    model_input = {c: reading.get(c, 0.0) for c in feature_columns}
    input_df = pd.DataFrame([model_input])[feature_columns]
    input_scaled = scaler.transform(input_df)

    reading["prediction"] = int(model.predict(input_scaled)[0])
    reading["probability_maintenance"] = float(model.predict_proba(input_scaled)[0][1])

    st.session_state.rt_data = pd.concat(
        [st.session_state.rt_data, pd.DataFrame([reading])], ignore_index=True
    ).tail(300)

    if missing_cols:
        st.warning(
            f"Kolom berikut tidak tersedia dari sensor dan diisi 0: {', '.join(missing_cols)}"
        )

rt_df = st.session_state.rt_data

if rt_df.empty:
    st.info("Menunggu data pertama... aktifkan 'Stream aktif' jika belum menyala.")
else:
    latest = rt_df.iloc[-1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Speed (RPM)", f"{latest.machine_speed_rpm:.0f}")
    m2.metric("Temp (°C)", f"{latest.temperature_c:.1f}")
    m3.metric("Humidity (%)", f"{latest.humidity_percent:.1f}")
    m4.metric("Vibration", f"{latest.vibration_level:.3f}")
    m5.metric("Energy (kWh)", f"{latest.energy_usage_kwh:.1f}")

    st.divider()
    proba = latest.probability_maintenance
    if latest.prediction == 1:
        st.error(f"⚠️ Mesin diprediksi **BUTUH MAINTENANCE** (probabilitas {proba:.1%})")
    else:
        st.success(f"✅ Mesin diprediksi **AMAN** (probabilitas butuh maintenance {proba:.1%})")
    st.progress(float(proba))

    if proba >= threshold:
        st.toast(
            f"⚠️ Alert: probabilitas maintenance {proba:.1%} melewati threshold {threshold:.0%}",
            icon="⚠️",
        )

    st.subheader("Grafik live")
    chart_df = rt_df.set_index("timestamp")
    st.line_chart(chart_df[["probability_maintenance"]])
    st.line_chart(chart_df[["machine_speed_rpm"]])
    st.line_chart(chart_df[["temperature_c", "humidity_percent"]])
    st.line_chart(chart_df[["vibration_level"]])
    st.line_chart(chart_df[["energy_usage_kwh"]])

    st.subheader("Data terbaru")
    st.dataframe(rt_df.tail(20), use_container_width=True)

    csv_result = rt_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download data sesi ini (CSV)",
        data=csv_result,
        file_name="realtime_predictions.csv",
        mime="text/csv",
        key="rt_download",
    )

st.divider()
st.caption("COMPFEST 18 AIC — Predictive Maintenance untuk Industri Tekstil")
