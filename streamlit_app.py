import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Streamlit App Configuration
# ============================================================

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="⏱️",
    layout="centered",
)


# ============================================================
# 2. Load Trained Model
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "delivery_time_model.pkl",
)


@st.cache_resource
def load_model():
    """Load the trained delivery-time prediction pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "delivery_time_model.pkl was not found. "
            "Place the trained model file in the same folder as streamlit_app.py."
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "predict"):
        raise TypeError(
            "delivery_time_model.pkl does not contain a valid prediction model/pipeline."
        )

    return loaded_model


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ============================================================
# 3. UI Options
# ============================================================

DROPDOWN_OPTIONS = {
    "Traffic_Level": ["Low", "Medium", "High", "Jam"],
    "Weather": ["Clear", "Windy", "Foggy", "Rain", "Storm"],
    "Vehicle_Type": ["Bike", "Scooter", "Electric Bike", "Car"],
    "Day_of_Week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    "Cuisine_Type": [
        "Fast Food",
        "Indian",
        "Italian",
        "Chinese",
        "Desserts",
        "Healthy",
    ],
}


# ============================================================
# 4. Domain Multipliers
# ============================================================

TRAFFIC_MULTIPLIERS = {
    "Low": 1.0,
    "Medium": 1.25,
    "High": 1.65,
    "Jam": 2.2,
}

WEATHER_MULTIPLIERS = {
    "Clear": 1.0,
    "Windy": 1.1,
    "Foggy": 1.25,
    "Rain": 1.45,
    "Storm": 1.85,
}

# Retained from the supplied Flask application.
VEHICLE_SPEED_FACTORS = {
    "Electric Bike": 0.9,
    "Scooter": 1.0,
    "Bike": 1.15,
    "Car": 1.35,
}


# ============================================================
# 5. Required Model Features
# ============================================================

REQUIRED_COLUMNS = [
    "Road_Distance_km",
    "Average_Speed_kmph",
    "Traffic_Level",
    "Day_of_Week",
    "Preparation_Time_Min",
    "Dropoff_Zone",
    "Vehicle_Type",
    "Order_Hour",
    "Restaurant_Load",
    "Is_Weekend",
    "Weather",
    "Cuisine_Type",
    "Pickup_Zone",
    "Is_Festival",
    "Number_of_Signals",
    "Rider_Experience_Years",
    "Delivery_Priority",
    "Rider_Rating",
    "Delivery_Distance_Category",
    "Order_Items",
    "Restaurant_Rating",
    "Estimated_Travel_Time",
]


# ============================================================
# 6. Default Values for Features Not Exposed in the UI
# ============================================================

DEFAULT_VALUES = {
    "Preparation_Time_Min": 15,
    "Dropoff_Zone": "Zone_A",
    "Order_Hour": 14,
    "Restaurant_Load": "Medium",
    "Cuisine_Type": "Fast Food",
    "Pickup_Zone": "Zone_A",
    "Is_Festival": 0,
    "Number_of_Signals": 3,
    "Rider_Experience_Years": 3,
    "Delivery_Priority": "Normal",
    "Rider_Rating": 4.5,
    "Delivery_Distance_Category": "Medium",
    "Order_Items": 2,
    "Restaurant_Rating": 4.2,
}


# ============================================================
# 7. Feature Preparation
# ============================================================

def prepare_features(input_data):
    """Prepare user inputs in the same structure expected by the trained pipeline."""

    df = pd.DataFrame([input_data])

    # Fill baseline values for features not exposed by the UI.
    for col, default in DEFAULT_VALUES.items():
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = default

    traffic = df["Traffic_Level"].iloc[0]
    weather = df["Weather"].iloc[0]
    vehicle = df["Vehicle_Type"].iloc[0]

    t_mult = TRAFFIC_MULTIPLIERS.get(traffic, 1.0)
    w_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
    v_mult = VEHICLE_SPEED_FACTORS.get(vehicle, 1.0)

    # Base travel time.
    base_time_min = (
        df["Road_Distance_km"]
        / np.maximum(df["Average_Speed_kmph"], 1)
    ) * 60

    # Same domain feature calculation used by the Flask application.
    df["Estimated_Travel_Time"] = (
        base_time_min * t_mult * w_mult * v_mult
    )

    # Preparation-time adjustment during extreme delay conditions.
    if traffic in ["High", "Jam"] or weather in ["Rain", "Storm"]:
        df["Preparation_Time_Min"] = (
            df["Preparation_Time_Min"] * 1.3
        )

    # Ensure every model feature exists and preserve the expected order.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[REQUIRED_COLUMNS]


# ============================================================
# 8. Application UI
# ============================================================

st.title("⏱️ Food Delivery Time Predictor")
st.caption(
    "Food-delivery time prediction by Shaik Afzal Hussain"
)

st.markdown(
    """
    Enter the delivery conditions below and the trained
    Model will estimate the delivery time.
    """
)

with st.form("delivery_prediction_form"):

    st.subheader("Delivery Information")

    col1, col2 = st.columns(2)

    with col1:
        road_distance = st.number_input(
            "Road Distance (km)",
            min_value=0.1,
            value=2.4,
            step=0.1,
        )

    with col2:
        average_speed = st.number_input(
            "Average Speed (km/h)",
            min_value=1.0,
            value=40.0,
            step=0.1,
        )

    col3, col4 = st.columns(2)

    with col3:
        traffic_level = st.selectbox(
            "Traffic Level",
            DROPDOWN_OPTIONS["Traffic_Level"],
            index=1,
        )

    with col4:
        weather = st.selectbox(
            "Weather Condition",
            DROPDOWN_OPTIONS["Weather"],
            index=0,
        )

    col5, col6 = st.columns(2)

    with col5:
        vehicle_type = st.selectbox(
            "Vehicle Type",
            DROPDOWN_OPTIONS["Vehicle_Type"],
            index=1,
        )

    with col6:
        day_of_week = st.selectbox(
            "Day of Week",
            DROPDOWN_OPTIONS["Day_of_Week"],
            index=0,
        )

    predict_button = st.form_submit_button(
        "⚡ Calculate Estimated Time",
        use_container_width=True,
    )


# ============================================================
# 9. Prediction
# ============================================================

if predict_button:

    try:
        user_inputs = {
            "Road_Distance_km": float(road_distance),
            "Average_Speed_kmph": float(average_speed),
            "Traffic_Level": traffic_level,
            "Weather": weather,
            "Vehicle_Type": vehicle_type,
            "Day_of_Week": day_of_week,
        }

        user_inputs["Is_Weekend"] = (
            1
            if day_of_week in ["Saturday", "Sunday"]
            else 0
        )

        input_df = prepare_features(user_inputs)

        prediction = model.predict(input_df)
        result = round(float(prediction[0]), 2)

        st.success("Prediction generated successfully.")

        st.markdown(
            f"""
            <div style="
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                background: linear-gradient(
                    135deg,
                    rgba(2,132,199,0.18),
                    rgba(56,189,248,0.10)
                );
                border: 1px solid rgba(56,189,248,0.7);
                margin-top: 20px;
            ">
                <div style="
                    font-size: 14px;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">
                    Estimated Arrival Time
                </div>
                <div style="
                    font-size: 36px;
                    font-weight: 700;
                    color: #38bdf8;
                    margin-top: 6px;
                ">
                    {result} mins
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    except Exception as e:
        st.error(
            f"Error processing prediction: {e}"
        )


# ============================================================
# 10. Footer
# ============================================================

st.divider()

st.caption(
    "Food Delivery Time Prediction • "
    "Gradient Boosting Regression • Shaik Afzal Hussain • Streamlit"
)