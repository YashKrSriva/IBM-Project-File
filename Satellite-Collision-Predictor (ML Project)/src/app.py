import streamlit as st
import pandas as pd
import joblib
import pyttsx3

# Streamlit UI
st.set_page_config(page_title="Satellite Collision Risk Predictor", layout="centered")
st.title("🛰 Satellite Collision Risk Predictor")

# Load model + preprocessor
model, preprocessor = joblib.load("models/sat_rf_model.pkl")

st.markdown("Enter approach parameters and click **Predict**")

# Inputs
dist = st.number_input("Relative Distance (km)", min_value=0.0, value=3.0, step=0.1)
speed = st.number_input("Relative Speed (km/s)", min_value=0.0, value=10.0, step=0.1)
size = st.number_input("Debris Size (cm)", min_value=0.1, value=15.0, step=0.1)
angle = st.slider("Approach Angle (degrees)", min_value=0, max_value=90, value=40)
dtype = st.selectbox("Debris Type", ["Metal", "Rock", "Fragment"])


# PREDICTION + VOICE ENGINE
if st.button("Predict"):

    engine = pyttsx3.init()
    engine.setProperty('rate', 170)       
    engine.setProperty('volume', 1.0)      

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  

    # for input
    input_df = pd.DataFrame({
        'Relative_Distance':[dist],
        'Relative_Speed':[speed],
        'Debris_Size':[size],
        'Approach_Angle':[angle],
        'Debris_Type':[dtype]
    })

    # Predict
    X_in = preprocessor.transform(input_df)
    pred = model.predict(X_in)[0]
    prob = model.predict_proba(X_in)[0][1]

    # Result messages
    if pred == 1:
        result_text = "⚠ HIGH COLLISION RISK"
        speak_text = "Warning. High collision risk detected. Take immediate action."
        st.error(result_text)
    else:
        result_text = "✔ SAFE — Low collision risk"
        speak_text = "Safe. No collision danger detected."
        st.success(result_text)

    # Display probability
    st.write(f"Risk Probability: {prob:.2f}")

    # Speak the result
    try:
        engine.say(speak_text)
        engine.runAndWait()
    except:
        pass
