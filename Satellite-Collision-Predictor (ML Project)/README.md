**🛰 Satellite Collision Risk Predictor**

A simple machine learning project that helps predict whether a satellite is at risk of colliding with space debris.

This project uses a small dataset with satellite approach parameters and builds an ML model that classifies the situation as:

* Safe (0)

* High Collision Risk (1)


The model is deployed through a Streamlit web app where users can enter values and instantly see the prediction — with optional voice alerts.

**What This Project Does**

* Predicts collision risk based on distance, speed, debris size, approach angle, and type of debris

* Uses Random Forest Classifier as the final model

* Includes a clean Streamlit UI for easy testing

* Outputs risk probability and a voice-based alert

* Entire model and preprocessor are saved and loaded for real-time predictions

**Dataset**

* The dataset contains the following features:

* Relative Distance (km)

* Relative Speed (km/s)

* Debris Size (cm)

* Approach Angle (degrees)

* Debris Type (Metal / Rock / Fragment)

* Risk Label (0 or 1)

This is a synthetic dataset created for learning purposes.

Machine Learning Workflow

1. Load and preprocess data


2. Encode categorical feature (Debris Type)


3. Split into train & test sets


4. Train Logistic Regression and Random Forest


5. Evaluate both models


6. Save the best model (Random Forest)


7. Deploy it using Streamlit

**Features**

* Clean and simple UI

* Real-time prediction

* Risk probability shown clearly

* Voice alert when prediction is made

* Easy to modify or extend

**What I want to improve in future**

* Use a bigger, more realistic dataset

* Pull real debris-tracking data from NASA APIs

* Add orbit visualizations

* Deploy the app online

**Author**
* Yash Kumar Srivastava
