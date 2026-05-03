import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Page setup 
st.set_page_config(page_title="Heart Checker", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details to check heart disease risk")

# Load dataset
data = pd.read_csv("heart.csv")

X = data[['Age', 'BP', 'Cholesterol']]
y = data['Disease']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Input UI
st.markdown("### 🧑 Patient Information")

name = st.text_input("Patient Name")

age = st.number_input("Age", 1, 100, 40)
bp = st.number_input("Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)

# Button
if st.button("Check Heart Risk ❤️"):

    prediction = model.predict([[age, bp, chol]])

    st.markdown("---")

    if name == "":
        st.warning("Please enter patient name first")
    else:
        st.subheader(f"Patient: {name}")

        if prediction[0] == 1:
            st.error(f"⚠️ {name} has HIGH risk of heart disease")
            st.write("Advice: Consult doctor, reduce stress, healthy diet")
        else:
            st.success(f"✅ {name} has LOW risk of heart disease")
            st.write("Advice: Keep maintaining healthy lifestyle")

# Footer
st.markdown("---")
st.caption("Mini Data Science Project | Heart Disease Prediction System")