from flask import Flask, jsonify, render_template_string, request
import os
import joblib
import numpy as np
import pandas as pd

# 1. Initialize Flask App
app = Flask(__name__)

# Load the saved model pipeline
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delivery_time_model.pkl")
model = joblib.load(MODEL_PATH)

if not hasattr(model, "predict"):
    raise TypeError("delivery_time_model.pkl does not contain a valid prediction model/pipeline.")

# Categorical options for the UI dropdowns
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

# Domain Multipliers to enforce real-world delay sensitivity
# NOTE: Vehicle factors are retained exactly as supplied; they are business logic.
TRAFFIC_MULTIPLIERS = {"Low": 1.0, "Medium": 1.25, "High": 1.65, "Jam": 2.2}

WEATHER_MULTIPLIERS = {
    "Clear": 1.0,
    "Windy": 1.1,
    "Foggy": 1.25,
    "Rain": 1.45,
    "Storm": 1.85,
}

VEHICLE_SPEED_FACTORS = {
    "Electric Bike": 0.9,
    "Scooter": 1.0,
    "Bike": 1.15,
    "Car": 1.35,
}

# All features expected by your trained ColumnTransformer / Pipeline
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

# Baseline defaults for unexposed features
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


def prepare_features(input_data):
    """Dynamically applies physical traffic and weather delay factors before prediction."""
    df = pd.DataFrame([input_data])

    # Fill default baseline values
    for col, default in DEFAULT_VALUES.items():
        if col not in df.columns or df[col].iloc[0] is None:
            df[col] = default

    traffic = df["Traffic_Level"].iloc[0] if "Traffic_Level" in df else "Medium"
    weather = df["Weather"].iloc[0] if "Weather" in df else "Clear"
    vehicle = df["Vehicle_Type"].iloc[0] if "Vehicle_Type" in df else "Scooter"

    t_mult = TRAFFIC_MULTIPLIERS.get(traffic, 1.0)
    w_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
    v_mult = VEHICLE_SPEED_FACTORS.get(vehicle, 1.0)

    # Base travel time
    base_time_min = (
        df["Road_Distance_km"] / np.maximum(df["Average_Speed_kmph"], 1)
    ) * 60

    # Dynamic travel time scaling
    df["Estimated_Travel_Time"] = base_time_min * t_mult * w_mult * v_mult

    # Prep adjustment during extreme delay factors
    if traffic in ["High", "Jam"] or weather in ["Rain", "Storm"]:
        df["Preparation_Time_Min"] = df["Preparation_Time_Min"] * 1.3

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[REQUIRED_COLUMNS]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delivery Time Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --input-bg: #334155;
            --border-color: #475569;
        }

        .light-theme {
            --bg-body: #f1f5f9;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --accent-blue: #0284c7;
            --accent-glow: rgba(2, 132, 199, 0.15);
            --input-bg: #f8fafc;
            --border-color: #cbd5e1;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            transition: all 0.3s ease;
        }

        .container {
            max-width: 520px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }

        h2 {
            margin: 0;
            font-size: 22px;
            font-weight: 700;
        }

        .theme-btn {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-sub);
            margin-bottom: 6px;
            margin-top: 14px;
        }

        input, select {
            width: 100%;
            padding: 12px;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 14px;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        button.submit-btn {
            width: 100%;
            margin-top: 24px;
            padding: 14px;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            border: none;
            color: white;
            font-weight: 700;
            font-size: 15px;
            border-radius: 10px;
            cursor: pointer;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        button.submit-btn:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 24px;
            padding: 20px;
            background: var(--accent-glow);
            border: 1px solid var(--accent-blue);
            border-radius: 12px;
            text-align: center;
        }

        .result-title {
            font-size: 13px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>⏱️ Delivery Time Predictor</h2>
            <button class="theme-btn" onclick="toggleTheme()">☀️ / 🌙</button>
        </div>

        <form action="/predict_ui" method="POST">
            <div class="grid-2">
                <div>
                    <label>Road Distance (km):</label>
                    <input type="number" step="0.1" name="Road_Distance_km" required value="{{ inputs.get('Road_Distance_km', '2.4') }}">
                </div>
                <div>
                    <label>Average Speed (km/h):</label>
                    <input type="number" step="0.1" name="Average_Speed_kmph" required value="{{ inputs.get('Average_Speed_kmph', '40.0') }}">
                </div>
            </div>

            <div class="grid-2">
                <div>
                    <label>Traffic Level:</label>
                    <select name="Traffic_Level">
                        {% for option in options['Traffic_Level'] %}
                            <option value="{{ option }}" {% if inputs.get('Traffic_Level') == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label>Weather Condition:</label>
                    <select name="Weather">
                        {% for option in options['Weather'] %}
                            <option value="{{ option }}" {% if inputs.get('Weather') == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="grid-2">
                <div>
                    <label>Vehicle Type:</label>
                    <select name="Vehicle_Type">
                        {% for option in options['Vehicle_Type'] %}
                            <option value="{{ option }}" {% if inputs.get('Vehicle_Type') == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label>Day of Week:</label>
                    <select name="Day_of_Week">
                        {% for option in options['Day_of_Week'] %}
                            <option value="{{ option }}" {% if inputs.get('Day_of_Week') == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <button type="submit" class="submit-btn">⚡ Calculate Estimated Time</button>
        </form>

        {% if prediction %}
            <div class="result-card">
                <div class="result-title">Estimated Arrival Time</div>
                <div class="result-value">{{ prediction }} mins</div>
            </div>
        {% endif %}
    </div>

    <script>
        function toggleTheme() {
            document.body.classList.toggle('light-theme');
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE, options=DROPDOWN_OPTIONS, inputs={}
    )


@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        user_inputs = {
            "Road_Distance_km": float(request.form.get("Road_Distance_km")),
            "Average_Speed_kmph": float(request.form.get("Average_Speed_kmph")),
            "Traffic_Level": request.form.get("Traffic_Level"),
            "Weather": request.form.get("Weather"),
            "Vehicle_Type": request.form.get("Vehicle_Type"),
            "Day_of_Week": request.form.get("Day_of_Week"),
        }

        user_inputs["Is_Weekend"] = (
            1
            if user_inputs["Day_of_Week"] in ["Saturday", "Sunday"]
            else 0
        )

        input_df = prepare_features(user_inputs)

        prediction = model.predict(input_df)
        result = round(float(prediction[0]), 2)

        return render_template_string(
            HTML_TEMPLATE,
            options=DROPDOWN_OPTIONS,
            inputs=user_inputs,
            prediction=result,
        )
    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = prepare_features(data)

        prediction = model.predict(input_df)
        return jsonify(
            {
                "status": "success",
                "predicted_delivery_time_min": float(
                    np.round(prediction[0], 2)
                ),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False)