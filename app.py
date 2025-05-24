import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the symptom list
with open("symptoms_list.pkl", "rb") as f:
    symptoms_list = pickle.load(f)

# Title
st.title("🩺 Disease Predictor from Symptoms")

# Instruction
st.write("Select symptoms you are experiencing:")

# Multi-select input
selected_symptoms = st.multiselect("Symptoms", symptoms_list)

def symptoms_to_vector(user_symptoms, all_symptoms):
    return [1 if s in user_symptoms else 0 for s in all_symptoms]

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = [symptoms_to_vector(selected_symptoms, symptoms_list)]
        prediction = model.predict(input_vector)[0]
        st.success(f"🧾 Predicted Disease: **{prediction}**")
