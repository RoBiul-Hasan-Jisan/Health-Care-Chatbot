

import streamlit as st
import pandas as pd
import joblib
import spacy


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path   = os.path.join(BASE_DIR, "model", "health_model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
desc_path    = os.path.join(BASE_DIR, "model", "symptom_description.csv")
prec_path    = os.path.join(BASE_DIR, "model", "precaution.csv")

model = joblib.load(model_path)
le = joblib.load(encoder_path)
desc_df = pd.read_csv(desc_path)
prec_df = pd.read_csv(prec_path)

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """Use spaCy to tokenize and clean text."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

def match_symptoms(text_tokens, symptom_columns):
    """Convert user input into binary feature vector for model."""
    features = [1 if sym in text_tokens else 0 for sym in symptom_columns]
    return [features]

def get_description(disease_name):
    """Return disease description from CSV."""
    row = desc_df[desc_df['disease'].str.lower() == disease_name.lower()]
    return row['description'].values[0] if not row.empty else "Description not available."

def get_precautions(disease_name):
    """Return disease precautions from CSV."""
    row = prec_df[prec_df['disease'].str.lower() == disease_name.lower()]
    return row['precaution'].values[0] if not row.empty else "Precautions not available."


st.set_page_config(page_title="Health Care Chatbot", page_icon="🩺")
st.title(" Health-Care Chatbot (NLP + ML)")
st.write("Enter your symptoms in plain text, e.g., 'I have a headache and vomiting'")

user_input = st.text_area("Your Symptoms:")

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms!")
    else:
        # Preprocess text
        tokens = preprocess_text(user_input)
        # Match symptoms with model columns
        symptom_columns = model.feature_names_in_
        X_input = match_symptoms(tokens, symptom_columns)
        # Predict
        pred_encoded = model.predict(X_input)
        disease_name = le.inverse_transform(pred_encoded)[0]

        # Get description and precautions
        description = get_description(disease_name)
        precautions = get_precautions(disease_name)

        # Display results
        st.success(f"**Predicted Disease:** {disease_name}")
        st.info(f"**Description:** {description}")
        st.warning(f"**Precautions:** {precautions}")
