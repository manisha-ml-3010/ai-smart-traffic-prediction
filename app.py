
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="AI Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart Traffic Prediction System")
st.write("Predict traffic volume using Machine Learning.")

# Load dataset
data = pd.read_csv("data/traffic.csv")

# Load or train model automatically
if os.path.exists("models/traffic_model.pkl"):
    model = joblib.load("models/traffic_model.pkl")
else:
    X = data.drop("traffic_volume", axis=1)
    y = data["traffic_volume"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/traffic_model.pkl")

st.subheader("Enter Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Select Hour", 0, 23, 8)
    day = st.selectbox(
        "Select Day",
        [1, 2, 3, 4, 5, 6, 7]
    )

with col2:
    temperature = st.slider(
        "Temperature (°C)", 0, 50, 28
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Cloudy", "Rainy"]
    )

weather_code = {
    "Clear": 0,
    "Cloudy": 1,
    "Rainy": 2
}

if st.button("Predict Traffic"):
    input_data = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day,
        "temperature": temperature,
        "weather_code": weather_code[weather]
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Traffic Volume",
        f"{prediction:.0f}"
    )

    if prediction < 500:
        st.success("Low Traffic 🟢")
    elif prediction < 850:
        st.warning("Moderate Traffic 🟡")
    else:
        st.error("High Traffic 🔴")

st.divider()

st.subheader("Traffic Data Visualization")

st.line_chart(
    data.set_index("hour")["traffic_volume"]
)

st.caption(
    "Educational ML prototype using illustrative sample data."
      )
