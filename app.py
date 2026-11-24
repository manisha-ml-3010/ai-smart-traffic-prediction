
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Traffic Management",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart Traffic Management System")
st.write("Predict traffic volume using Machine Learning")

# Load data and model
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")
model = joblib.load("models/traffic_model.pkl")

# User inputs
st.sidebar.header("Traffic Conditions")

hour = st.sidebar.slider("Hour", 0, 23, 8)
day = st.sidebar.slider("Day of Week (0=Monday)", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 6)

temp = st.sidebar.slider(
    "Temperature (Kelvin)", 240.0, 310.0, 280.0
)
rain = st.sidebar.number_input("Rain (mm)", 0.0, 100.0, 0.0)
snow = st.sidebar.number_input("Snow (mm)", 0.0, 100.0, 0.0)
clouds = st.sidebar.slider("Cloud Coverage", 0, 100, 40)

weather = st.sidebar.selectbox(
    "Weather",
    sorted(df["weather_main"].dropna().unique())
)

# Prediction
if st.button("Predict Traffic"):
    input_data = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "month": month,
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "weather_main": weather
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("AI Prediction Result")
    st.metric("Predicted Traffic Volume", f"{prediction:.0f} vehicles")

    if prediction < 2000:
        st.success("Low Traffic")
        st.info("Normal traffic flow expected.")
    elif prediction < 4000:
        st.warning("Moderate Traffic")
        st.info("Monitor traffic conditions.")
    else:
        st.error("High Traffic")
        st.info("Consider reviewing traffic signal timing.")

# Historical analytics
st.subheader("Historical Traffic Analytics")

df["date_time"] = pd.to_datetime(df["date_time"])
hourly = df.groupby(df["date_time"].dt.hour)[
    "traffic_volume"
].mean()

st.line_chart(hourly)
st.subheader("Dataset Overview")
st.write(df.head())
