import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "health_model.pkl")
le_path    = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
desc_path  = os.path.join(BASE_DIR, "model", "symptom_description.csv")
prec_path  = os.path.join(BASE_DIR, "model", "precaution.csv")

model = joblib.load(model_path)
le = joblib.load(le_path)
desc_df = pd.read_csv(desc_path, quotechar='"')
prec_df = pd.read_csv(prec_path, quotechar='"')

feature_columns = model.feature_names_in_.tolist()


st.set_page_config(
    page_title="Health-Care Chatbot",
    page_icon="",
    layout="centered",
)


st.markdown("""
   <style>
    body {
        background: linear-gradient(135deg, #f0f8ff, #e0f7fa);
        font-family: 'Segoe UI', sans-serif;
    }

    .hero {
        background: #ffffffcc;
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00796b;
    }
    .hero p {
        font-size: 1rem;
        color: #004d40;
    }

    .stMultiSelect > div {
        background-color: #ffffffcc !important;
        border: 1px solid #00bfa5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    .stButton > button {
        background: linear-gradient(90deg, #00bfa5, #00796b);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00796b, #004d40);
        transform: scale(1.03);
    }

    .result-box {
        background: #ffffff;
        border-left: 6px solid #00bfa5;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00796b;
    }
    .desc, .prec {
        font-size: 1rem;
        color: #004d40;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
</style>

""", unsafe_allow_html=True)


st.markdown("""
    <div class="hero">
        <h1> Health-Care Chatbot</h1>
        <p>Your smart assistant for quick disease prediction </p>
    </div>
""", unsafe_allow_html=True)


selected_symptoms = st.multiselect(
    " Select Symptoms:",
    options=feature_columns,
    help="Start typing to quickly find symptoms."
)


if st.button(" Predict Disease"):
    if not selected_symptoms:
        st.warning(" Please select at least one symptom.")
    else:
        # Simulate thinking
        with st.spinner('Analyzing symptoms... '):
            time.sleep(1.2)

        # Create input vector
        input_vector = np.zeros(len(feature_columns), dtype=int)
        for symptom in selected_symptoms:
            if symptom in feature_columns:
                idx = feature_columns.index(symptom)
                input_vector[idx] = 1

        # Predict
        predicted_encoded = model.predict([input_vector])[0]
        predicted_disease = le.inverse_transform([predicted_encoded])[0]

        # Fetch info
        desc = desc_df.loc[desc_df['disease'] == predicted_disease, 'description'].values
        desc = desc[0] if len(desc) > 0 else "Description not found."

        precautions = prec_df.loc[prec_df['disease'] == predicted_disease, 'precaution'].values
        precautions = precautions[0] if len(precautions) > 0 else "Precautions not found."

        # Show results
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title"> Predicted Disease: <b>{predicted_disease}</b></div>
                <div class="desc"><b>Description:</b> {desc}</div>
                <div class="prec"><b>Precautions:</b> {precautions}</div>
            </div>
        """, unsafe_allow_html=True)
