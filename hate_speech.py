from profanity_check import predict

def detect_hate_speech(text: str) -> str:
    try:
        result = predict([text])[0]
        if result == 1:
            return "⚠️ Offensive or inappropriate content detected."
        else:
            return "✅ No hate speech detected."
    except Exception as e:
        return f"Error detecting hate speech: {str(e)}"
