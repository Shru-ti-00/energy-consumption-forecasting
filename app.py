import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

model = joblib.load("xgboost_energy_model.pkl")

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("⚡ Energy Consumption Forecasting System")

st.write(
    "Predict Zone 1 Power Consumption using "
    "weather and time-series features."
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header("About Project")

st.sidebar.write("""
This project uses XGBoost Machine Learning
for energy forecasting using:

- Weather Features
- Time Features
- Lag Features
- Rolling Mean Features
- Interaction Features
""")

# -----------------------------------
# USER INPUTS
# -----------------------------------

st.header("📥 Enter Input Details")

# WEATHER FEATURES

temperature = st.slider(
    "Temperature",
    0.0,
    50.0,
    25.0
)

humidity = st.slider(
    "Humidity",
    0.0,
    100.0,
    50.0
)

windspeed = st.slider(
    "Wind Speed",
    0.0,
    20.0,
    3.0
)

general_diffuse_flows = st.number_input(
    "General Diffuse Flows",
    min_value=0.0,
    value=200.0
)

diffuse_flows = st.number_input(
    "Diffuse Flows",
    min_value=0.0,
    value=100.0
)

# TIME FEATURES

hour = st.slider(
    "Hour",
    0,
    23,
    12
)

day = st.slider(
    "Day",
    1,
    31,
    15
)

month = st.slider(
    "Month",
    1,
    12,
    6
)

day_of_week = st.slider(
    "Day Of Week (0=Monday, 6=Sunday)",
    0,
    6,
    2
)

day_of_year = st.slider(
    "Day Of Year",
    1,
    365,
    150
)

week_of_year = st.slider(
    "Week Of Year",
    1,
    52,
    24
)

is_weekend = st.selectbox(
    "Is Weekend?",
    [0, 1]
)

# LAG FEATURES

lag_1 = st.number_input(
    "Previous Hour Consumption (Lag_1)",
    min_value=0.0,
    value=30000.0
)

lag_24 = st.number_input(
    "Previous Day Consumption (Lag_24)",
    min_value=0.0,
    value=29000.0
)

# ROLLING FEATURES

rolling_mean_3 = st.number_input(
    "Rolling Mean 3",
    min_value=0.0,
    value=29500.0
)

rolling_mean_24 = st.number_input(
    "Rolling Mean 24",
    min_value=0.0,
    value=28500.0
)

# -----------------------------------
# INTERACTION FEATURES
# -----------------------------------

temp_humidity = temperature * humidity

temp_wind = temperature * windspeed

# -----------------------------------
# CREATE INPUT DATAFRAME
# -----------------------------------

input_data = pd.DataFrame({

    'Temperature': [temperature],

    'Humidity': [humidity],

    'WindSpeed': [windspeed],

    'GeneralDiffuseFlows': [general_diffuse_flows],

    'DiffuseFlows': [diffuse_flows],

    'Hour': [hour],

    'Day': [day],

    'Month': [month],

    'DayOfWeek': [day_of_week],

    'DayOfYear': [day_of_year],

    'WeekOfYear': [week_of_year],

    'IsWeekend': [is_weekend],

    'Lag_1': [lag_1],

    'Lag_24': [lag_24],

    'Rolling_Mean_3': [rolling_mean_3],

    'Rolling_Mean_24': [rolling_mean_24],

    'Temp_Humidity': [temp_humidity],

    'Temp_Wind': [temp_wind]

})

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("🔮 Predict Power Consumption"):

    prediction = model.predict(input_data)

    predicted_value = prediction[0]

    st.success(
        f"Predicted Zone 1 Power Consumption: "
        f"{predicted_value:.2f}"
    )

    # -----------------------------------
    # INSIGHTS
    # -----------------------------------

    st.subheader("📊 Consumption Analysis")

    if predicted_value > 40000:

        st.error(
            "⚠️ Very High Power Consumption"
        )

    elif predicted_value > 25000:

        st.warning(
            "⚡ Moderate Power Consumption"
        )

    else:

        st.success(
            "✅ Low Power Consumption"
        )

    # -----------------------------------
    # INPUT SUMMARY
    # -----------------------------------

    st.subheader("📋 Input Summary")

    st.dataframe(input_data)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Built using Streamlit + XGBoost"
)