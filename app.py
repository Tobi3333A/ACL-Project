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



# Preparing input
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Sport": [sport],
    "Affiliation": [affiliation],
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


for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]


# Scale and predict
scaled = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled, columns=input_df.columns)
prediction = model.predict(scaled_df)[0]


st.subheader(f"Predicted ACL Risk Score: {prediction:.2f}")

if prediction > 4:
    st.error("ACL risk too high! You're advised to contact a health professional. Do you want to be connected with a health proffesional?")
    st.link_button("Get connected", "https://tobi3333a.github.io/Final-Four-s-ACL-Injury-Website/health.html")
else:
    st.success("You're good to go. Do you still want to be connected with a health professional?")
    st.link_button("Get connected", "https://tobi3333a.github.io/Final-Four-s-ACL-Injury-Website/HTML-Files/health.html")

