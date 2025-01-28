import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('selected_depression_cols_df.csv')
X = df.drop('Depression', axis=1)
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = joblib.load("student_depression_predicting_model.pkl")

st.title("Student Depression Prediction App")

# input fields for features
Age = st.slider("Age", min_value=10, max_value=60, value=20, step=1)
AcademicPressure = st.slider("Academic Pressure (1 = Low, 5 = High)", min_value=1, max_value=5, value=3, step=1)
StudySatisfaction = st.slider("Study Satisfaction (1 = Very Unsatisfied, 5 = Very Satisfied)", min_value=1, max_value=5, value=3, step=1)
DietaryHabits = st.selectbox("Dietary Habits (1=Unhealthy, 2=Moderate, 3=Healthy)", options=[0, 1, 2])
SuicidalThoughts = st.radio("Have you ever had suicidal thoughts? (0=No, 1=Yes)", options=[0, 1])
StudyHours = st.slider("Study Hours Per Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
FinancialStress = st.slider("Financial Stress (1 = Low, 5 = High)", min_value=1, max_value=5, value=3)

# Button for making a prediction after input of the above variables
if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "Age": [Age],
            "Academic Pressure": [AcademicPressure],
            "Study Satisfaction": [StudySatisfaction],
            "Dietary Habits": [DietaryHabits],
            "Have you ever had suicidal thoughts ?": [SuicidalThoughts],
            "Study Hours": [StudyHours],
            "Financial Stress": [FinancialStress]     
        }
    )
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("The student has depression")
    else:
        st.success("The student does not have depression")