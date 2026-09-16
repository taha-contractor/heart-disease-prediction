import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "RF_model.pkl"
COLUMNS_PATH = MODEL_DIR / "columns.pkl"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #374151
        );
        color: white;
        margin-bottom: 30px;
    }
    .hero h1 {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 8px;
    }
    .hero p {
        font-size: 17px;
        color: #d1d5db;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 18px;
    }
    .result-card {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin: 25px 0;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .result-card h2 {
        font-size: 30px;
        margin-bottom: 10px;
    }
    .result-card p {
        font-size: 17px;
    }
    div.stButton > button {
        height: 52px;
        font-size: 17px;
        font-weight: 650;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RF_model.pkl not found at:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    columns = None
    if COLUMNS_PATH.exists():
        columns = joblib.load(COLUMNS_PATH)
    return model, columns

try:
    model, trained_columns = load_model()

except Exception as error:
    st.error("❌ Unable to load the trained model.")
    st.code(str(error))
    st.info(
        "Make sure RF_model.pkl and columns.pkl are inside "
        "the models folder."
    )
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>
            A Machine Learning based application that predicts
            heart disease using a Random Forest Classifier.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("🩺 About the Project")
    st.write(
        """
        This application uses a Random Forest Classifier
        to predict the presence of heart disease based on
        patient information.
        """
    )

    st.divider()
    st.subheader("📊 Input Features")
    st.write(
        """
        • Age
        • Sex
        • Chest Pain Type
        • Resting Blood Pressure
        • Cholesterol
        • Fasting Blood Sugar
        • Resting ECG
        • Maximum Heart Rate
        • Exercise Angina
        • Oldpeak
        • ST Slope
        """
    )

    st.divider()
    st.subheader("🤖 Model")
    st.write("Random Forest Classifier")
    st.divider()
    st.caption(
        "⚠️ This application is developed for "
        "academic/educational purposes only."
    )

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2, gap="large")

with left:
    age = st.number_input(
        "Age",
        min_value=28,
        max_value=77,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"],
        format_func=lambda x:
            "Male" if x == "M" else "Female"
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Atypical Angina (ATA)",
            "NAP": "Non-Anginal Pain (NAP)",
            "TA": "Typical Angina (TA)",
            "ASY": "Asymptomatic (ASY)"
        }[x]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=0,
        max_value=603,
        value=200,
        step=1
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x:
            "No (0)" if x == 0 else "Yes (1)"
    )

with right:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy (LVH)"
        }[x]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=202,
        value=150,
        step=1
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"],
        format_func=lambda x:
            "No (N)" if x == "N" else "Yes (Y)"
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-2.6,
        max_value=6.2,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise relative to rest."
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Upsloping (Up)",
            "Flat": "Flat",
            "Down": "Downsloping (Down)"
        }[x]
    )

st.divider()
_, button_col, _ = st.columns([1, 2, 1])

with button_col:
    predict = st.button(
        "🔍 Predict Heart Disease",
        type="primary",
        use_container_width=True
    )

if predict:
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        }
    )

    try:
        if trained_columns is not None:
            trained_columns = list(trained_columns)
            input_data = input_data[trained_columns]

        prediction = model.predict(input_data)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                input_data
            )[0]

        st.markdown(
            '<div class="section-title">📋 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if int(prediction) == 1:
            st.markdown(
                """
                <div class="result-card">

                    <h2>⚠️ Heart Disease Predicted</h2>

                    <p>
                        The Random Forest model predicts
                        a positive heart disease status.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-card">

                    <h2>✅ No Heart Disease Predicted</h2>

                    <p>
                        The Random Forest model predicts
                        a negative heart disease status.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        if probability is not None:
            no_disease = float(probability[0]) * 100
            disease = float(probability[1]) * 100
            st.markdown(
                '<div class="section-title">'
                '📈 Prediction Probability'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "No Heart Disease",
                    f"{no_disease:.2f}%"
                )

            with col2:
                st.metric(
                    "Heart Disease",
                    f"{disease:.2f}%"
                )

            st.write("Heart Disease Probability")
            st.progress(
                max(
                    0.0,
                    min(1.0, disease / 100)
                )
            )

        with st.expander("👁️ View Entered Patient Information"):

            display_data = input_data.copy()

            display_data.columns = [
                "Age",
                "Sex",
                "Chest Pain Type",
                "Resting Blood Pressure",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Resting ECG",
                "Maximum Heart Rate",
                "Exercise Angina",
                "Oldpeak",
                "ST Slope"
            ]

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            """
            This prediction is intended only for educational
            purposes. It should not be considered a medical
            diagnosis or a substitute for professional medical
            advice.
            """
        )

    except Exception as error:
        st.error("❌ Prediction failed.")
        st.exception(error)

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

        <p>
            ❤️ Heart Disease Prediction
        </p>

        <p style="font-size:13px;color:gray;">
            Machine Learning Project • Random Forest Classifier
        </p>

    </div>
    """,
    unsafe_allow_html=True
)