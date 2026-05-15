# ==============================
# 🎤 VOICE TEST
# ==============================

import speech_recognition as sr

# =====================================================
# CREATE RECOGNIZER
# =====================================================

r = sr.Recognizer()

# =====================================================
# START MICROPHONE
# =====================================================

with sr.Microphone() as source:

    print("🎤 Speak Something...")

    audio = r.listen(source)

# =====================================================
# CONVERT SPEECH TO TEXT
# =====================================================

try:

    text = r.recognize_google(audio)

    print("\n✅ You Said:")
    print(text)

except:

    print("\n❌ Could not recognize voice")