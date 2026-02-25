import pandas as pd
import joblib
import streamlit as st

# Load model and scaler
model = joblib.load("acl_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏃‍♂️ ACL Risk Score Predictor")

# Inputs
age = st.number_input("Age", 13, 25)
sex = st.selectbox("Sex", ["Male", "Female"])
sport = st.selectbox("Sport", ["Basketball", "Football", "Soccer", "Lacrosse", "Gymnastics", "Other"])
affiliation = st.selectbox("Affiliation", ["School", "Club"])
bmi = st.number_input("BMI", 10, 40)
recovery_days = st.number_input("Recovery Days per Week", 0, 7)
training_hours = st.number_input("Training Hours per Week", 0, 40)
training_intensity = st.number_input("Training Intensity (1–5)", 1, 5)
match_count = st.number_input("Match Count per Week", 0, 10)
rest_days = st.number_input("Rest Between Events (days)", 0, 7)
load_balance = st.number_input("Load Balance Score (0–10)", 0, 10)
weight = st.number_input("Weight (kg)", 45, 105)
height = st.number_input("Height (cm)", 120, 220)

# Create base dataframe with numeric features
input_df = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Recovery_Days_Per_Week": [recovery_days],
    "Training_Hours_Per_Week": [training_hours],
    "Training_Intensity": [training_intensity],
    "Match_Count_Per_Week": [match_count],
    "Rest_Between_Events_Days": [rest_days],
    "Load_Balance_Score": [load_balance],
    "Weight_kg": [weight],
    "Height_cm": [height]
})

# One-hot encode Sex
input_df["Sex_Female"] = 1 if sex == "Female" else 0
input_df["Sex_Male"] = 1 if sex == "Male" else 0

# One-hot encode Sport (based on your options)
input_df["Sport_Basketball"] = 1 if sport == "Basketball" else 0
input_df["Sport_Football"] = 1 if sport == "Football" else 0
input_df["Sport_Soccer"] = 1 if sport == "Soccer" else 0
input_df["Sport_Lacrosse"] = 1 if sport == "Lacrosse" else 0
input_df["Sport_Gymnastics"] = 1 if sport == "Gymnastics" else 0
input_df["Sport_Other"] = 1 if sport == "Other" else 0

# One-hot encode Affiliation
input_df["Affiliation_Club"] = 1 if affiliation == "Club" else 0
input_df["Affiliation_School"] = 1 if affiliation == "School" else 0

# Now we need to know ALL the feature names that the model expects
# Since we don't have feature_names_in_, let's try to get them from the scaler or model
# If that fails, we'll need to define them manually

# Option 1: Try to get feature names from scaler (if available)
if hasattr(scaler, 'feature_names_in_'):
    expected_features = list(scaler.feature_names_in_)
else:
    # Option 2: Define them manually based on what was likely in training
    # This list should include ALL one-hot encoded columns
    expected_features = [
        "Age", "BMI", "Recovery_Days_Per_Week", "Training_Hours_Per_Week",
        "Training_Intensity", "Match_Count_Per_Week", "Rest_Between_Events_Days",
        "Load_Balance_Score", "Weight_kg", "Height_cm",
        "Sex_Female", "Sex_Male",
        "Sport_Basketball", "Sport_Football", "Sport_Soccer", 
        "Sport_Lacrosse", "Sport_Gymnastics", "Sport_Other",
        "Affiliation_Club", "Affiliation_School"
    ]

# Ensure all expected features are present in the correct order
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0

# Reorder columns to match expected features
input_df = input_df[expected_features]

# Scale and predict
scaled = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled, columns=input_df.columns)
prediction = model.predict(scaled_df)[0]

st.subheader(f"Predicted ACL Risk Score: {prediction:.2f}")

if prediction > 4:
    st.error("ACL risk too high! You're advised to contact a health professional. Do you want to be connected with a health professional?")
    st.link_button("Get connected", "https://tobi3333a.github.io/Final-Four-s-ACL-Injury-Website/health.html")
else:
    st.success("You're good to go. Do you still want to be connected with a health professional?")
    st.link_button("Get connected", "https://tobi3333a.github.io/Final-Four-s-ACL-Injury-Website/HTML-Files/health.html")