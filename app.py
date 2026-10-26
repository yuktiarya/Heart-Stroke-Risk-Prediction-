import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Health Risk Estimator",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        background: linear-gradient(125deg, #10243a 0%, #173d54 58%, #176b70 100%);
        color: #f8fafc;
    }
    .hero-eyebrow {
        margin: 0 0 0.6rem;
        color: #9fe8dc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.1;
    }
    .hero-copy {
        max-width: 680px;
        margin: 0.8rem 0 0;
        color: #d7e7ef;
        font-size: 1.03rem;
        line-height: 1.6;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }
    div.stButton > button {
        min-height: 3.1rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(90deg, #0f766e, #14958a);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        transition: filter 150ms ease, transform 150ms ease;
    }
    div.stButton > button:hover {
        border: 0;
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    .result-card {
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
        border-radius: 14px;
        background: rgba(15, 118, 110, 0.1);
        border: 1px solid rgba(15, 118, 110, 0.35);
    }
    .result-card h3 {
        margin: 0 0 0.35rem;
    }
    .result-card p {
        margin: 0;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model artifacts from the project directory.
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown(
    """
    <section class="hero">
        <p class="hero-eyebrow">Heart health · ML project</p>
        <h1>Heart Health Risk Estimator</h1>
        <p class="hero-copy">
            Enter the health details below to view the prediction from this
            machine-learning model.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("👤 Personal profile")
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        age = st.slider("Age", 18, 100, 40)
    with profile_col2:
        sex = st.selectbox("Sex", ["M", "F"], format_func=lambda value: "Male" if value == "M" else "Female")

with st.container(border=True):
    st.subheader("🩺 Clinical measurements")
    measure_col1, measure_col2, measure_col3 = st.columns(3)
    with measure_col1:
        resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
    with measure_col2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with measure_col3:
        max_hr = st.slider("Maximum heart rate", 60, 220, 150)

    measure_col4, measure_col5 = st.columns(2)
    with measure_col4:
        fasting_bs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
    with measure_col5:
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)

with st.container(border=True):
    st.subheader("📋 Symptoms & test results")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    with detail_col1:
        chest_pain = st.selectbox(
            "Chest pain type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda value: {
                "ATA": "Atypical angina",
                "NAP": "Non-anginal pain",
                "TA": "Typical angina",
                "ASY": "Asymptomatic",
            }[value],
        )
        exercise_angina = st.selectbox(
            "Exercise-induced angina",
            ["Y", "N"],
            format_func=lambda value: "Yes" if value == "Y" else "No",
        )
    with detail_col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    with detail_col3:
        st_slope = st.selectbox(
            "ST slope",
            ["Up", "Flat", "Down"],
            format_func=lambda value: {
                "Up": "Upsloping",
                "Flat": "Flat",
                "Down": "Downsloping",
            }[value],
        )

st.caption(
    "Educational demonstration only. This model is not a medical device and "
    "does not provide a diagnosis or replace advice from a qualified clinician."
)

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict = st.button("✨  Estimate risk", use_container_width=True)

if predict:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.warning(
            "The model predicts a higher-risk result. Please discuss your "
            "health concerns with a qualified healthcare professional."
        )
    else:
        st.success(
            "The model predicts a lower-risk result. This does not rule out "
            "heart disease or replace professional medical advice."
        )


# Old Version

""""
import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Stroke Prediction by akarsh")
st.markdown("Provide the following details to check your heart stroke risk:")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
"""