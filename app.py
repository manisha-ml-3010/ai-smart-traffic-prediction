
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Page settings
st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart Traffic Management System")
st.write("Predict traffic volume using Machine Learning")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv(
        "data/Metro_Interstate_Traffic_Volume.csv"
    )
    data["date_time"] = pd.to_datetime(data["date_time"])

    data["hour"] = data["date_time"].dt.hour
    data["day"] = data["date_time"].dt.dayofweek
    data["month"] = data["date_time"].dt.month

    return data

df = load_data()

# Prepare model
@st.cache_resource
def train_model():
    features = [
        "hour", "day", "month", "temp",
        "rain_1h", "snow_1h", "clouds_all",
        "weather_main"
    ]

    data = df.dropna(subset=features + ["traffic_volume"])

    X = data[features]
    y = data["traffic_volume"]

    preprocessor = ColumnTransformer([
        (
            "weather",
            OneHotEncoder(handle_unknown="ignore"),
            ["weather_main"]
        )
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)

    return model

model = train_model()
st.subheader("📊 Traffic Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Average Traffic", f"{df['traffic_volume'].mean():,.0f}")
col3.metric("Maximum Traffic", f"{df['traffic_volume'].max():,.0f}")
col4.metric("Average Temperature", f"{df['temp'].mean():.1f} K")

# Sidebar inputs
st.sidebar.header("Traffic Conditions")

hour = st.sidebar.slider("Hour", 0, 23, 8)
day = st.sidebar.slider("Day (0 = Monday)", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 6)

temp = st.sidebar.slider(
    "Temperature (Kelvin)", 240.0, 310.0, 280.0
)

rain = st.sidebar.number_input(
    "Rain (mm)", min_value=0.0, value=0.0
)

snow = st.sidebar.number_input(
    "Snow (mm)", min_value=0.0, value=0.0
)

clouds = st.sidebar.slider(
    "Cloud Coverage", 0, 100, 40
)

weather_options = sorted(
    df["weather_main"].dropna().unique().tolist()
)

weather = st.sidebar.selectbox(
    "Weather Condition", weather_options
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

    st.metric(
        "Predicted Traffic Volume",
        f"{prediction:,.0f} vehicles"
    )

    if prediction < 2000:
        st.success("LOW TRAFFIC")
    elif prediction < 4000:
        st.warning("MODERATE TRAFFIC")
    else:
        st.error("HIGH TRAFFIC")

# Historical analytics
st.subheader("📊 Historical Traffic Analytics")

hourly = df.groupby("hour")["traffic_volume"].mean()

st.line_chart(hourly)

st.subheader("📋 Dataset Overview")

st.dataframe(df.head(10), use_container_width=True)

st.write("Total Records:", len(df))
    
