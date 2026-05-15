# ==============================
# 🚀 VOICE + AI SYSTEM
# ==============================

import speech_recognition as sr
import joblib
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# =====================================================
# LOAD DEPARTMENT MODEL
# =====================================================

tfidf = joblib.load(
    "saved_department_model/tfidf_vectorizer.pkl"
)

department_model = joblib.load(
    "saved_department_model/department_model.pkl"
)

print("✅ Department Model Loaded")

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

print("✅ Sentiment Model Loaded")

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
# FINAL AI PREDICTION
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
# VOICE INPUT
# =====================================================

r = sr.Recognizer()

with sr.Microphone() as source:

    print("\n🎤 Speak Your Complaint...\n")

    audio = r.listen(source)

# =====================================================
# SPEECH TO TEXT
# =====================================================

try:

    text = r.recognize_google(audio)

    print("✅ You Said:")
    print(text)

    # =================================================
    # AI PREDICTION
    # =================================================

    department, sentiment, priority = final_prediction(text)

    print("\n🚀 AI RESULT\n")

    print("📂 Department :", department)

    print("😊 Sentiment  :", sentiment)

    print("⚠️ Priority   :", priority)

except:

    print("\n❌ Could not recognize voice")