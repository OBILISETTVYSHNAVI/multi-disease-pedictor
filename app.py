import streamlit as st
from utils import predict_heart, predict_diabetes, predict_pneumonia

st.set_page_config(page_title="🩺 Multi-Disease Predictor", layout="centered")
st.title("🧬 Multi-Disease Predictor (ML + DL)")
st.markdown("Predict **Heart Disease**, **Diabetes**, and **Pneumonia** using medical data or X-rays.")

st.sidebar.header("Choose Prediction Type")
app_mode = st.sidebar.radio("Select", ["Heart Disease", "Diabetes", "Pneumonia"])

if app_mode == "Heart Disease":
    st.subheader("🫀 Heart Disease Prediction")
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.slider("Chest Pain Type (0–3)", 0, 3)
    trestbps = st.slider("Resting BP", 90, 200)
    chol = st.slider("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [1, 0])
    restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 70, 210)
    exang = st.selectbox("Exercise Induced Angina", [1, 0])
    oldpeak = st.slider("ST Depression", 0.0, 6.0)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    ca = st.slider("Number of Major Vessels Colored", 0, 4)
    thal = st.selectbox("Thalassemia (0–2)", [0, 1, 2])

    if st.button("Predict Heart Disease"):
        result = predict_heart([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal])
        st.success("🛑 Heart Disease Detected" if result == 1 else "✅ No Heart Disease")

elif app_mode == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")
    preg = st.slider("Pregnancies", 0, 15)
    glucose = st.slider("Glucose Level", 0, 200)
    bp = st.slider("Blood Pressure", 0, 140)
    skin = st.slider("Skin Thickness", 0, 100)
    insulin = st.slider("Insulin", 0, 900)
    bmi = st.slider("BMI", 0.0, 70.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.slider("Age", 10, 100)

    if st.button("Predict Diabetes"):
        result = predict_diabetes([preg, glucose, bp, skin, insulin, bmi, dpf, age])
        st.success("🛑 Diabetes Detected" if result == 1 else "✅ No Diabetes")

elif app_mode == "Pneumonia":
    st.subheader("🩻 Pneumonia Detection from X-ray")
    img_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        st.image(img_file, caption="Uploaded X-ray", use_column_width=True)
        if st.button("Analyze X-ray"):
            result = predict_pneumonia(img_file)
            st.success(f"Prediction: {result}")
