# ==============================
# 🚀 AI COMPLAINT ANALYZER
# ==============================

import streamlit as st
import joblib
import torch
import speech_recognition as sr

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Complaint Analyzer",
    layout="centered"
)

# =====================================================
# TITLE
# =====================================================

st.title("🚀 AI Complaint Analyzer")

st.write("Department + Sentiment Prediction System")

# =====================================================
# LOAD DEPARTMENT MODEL
# =====================================================

tfidf = joblib.load(
    "saved_department_model/tfidf_vectorizer.pkl"
)

department_model = joblib.load(
    "saved_department_model/department_model.pkl"
)

# =====================================================
# LOAD SENTIMENT MODEL
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    "saved_sentiment_model"
)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_sentiment_model"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

sentiment_model = sentiment_model.to(device)

# =====================================================
# SENTIMENT FUNCTION
# =====================================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    if pred == 1:
        return "Positive"
    else:
        return "Negative"

# =====================================================
# FINAL PREDICTION FUNCTION
# =====================================================

def final_prediction(text):

    text_clean = text.lower()

    text_tfidf = tfidf.transform([text_clean])

    department = department_model.predict(
        text_tfidf
    )[0]

    sentiment = predict_sentiment(text)

    if sentiment == "Negative":
        priority = "High"
    else:
        priority = "Normal"

    return department, sentiment, priority

# =====================================================
# 🎤 VOICE INPUT FUNCTION
# =====================================================

def get_voice_input():

    r = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Speak Now...")

        audio = r.listen(source)

    try:

        text = r.recognize_google(audio)

        st.success("✅ Voice Captured")

        return text

    except:

        st.error("❌ Could not recognize voice")

        return ""

# =====================================================
# SESSION STATE
# =====================================================

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# =====================================================
# TEXT AREA
# =====================================================

user_input = st.text_area(
    "Enter Complaint Text",
    value=st.session_state.user_input,
    height=150
)

# =====================================================
# UPDATE SESSION STATE
# =====================================================

st.session_state.user_input = user_input

# =====================================================
# 🎤 VOICE BUTTON
# =====================================================

if st.button("🎤 Speak Complaint"):

    voice_text = get_voice_input()

    if voice_text != "":

        st.session_state.user_input = voice_text

        st.rerun()

# =====================================================
# 🔥 PREDICT BUTTON
# =====================================================

if st.button("Predict"):

    final_text = st.session_state.user_input

    if final_text.strip() == "":

        st.warning("Please enter complaint text")

    else:

        department, sentiment, priority = final_prediction(
            final_text
        )

        st.success("Prediction Completed")

        st.subheader("📂 Department")
        st.write(department)

        st.subheader("😊 Sentiment")
        st.write(sentiment)

        st.subheader("⚠️ Priority")
        st.write(priority)