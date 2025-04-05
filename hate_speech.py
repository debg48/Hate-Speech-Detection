import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

LABEL_THRESHOLDS = {
    "toxic": 0.5,
    "insult": 0.5,
    "threat": 0.5,
    "obscene": 0.5,
    "identity_hate": 0.4,
    "severe_toxic": 0.3
}

def detect_hate_speech(text: str) -> str:
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=10
        )
        response.raise_for_status()
        predictions = response.json()[0]

        triggered_labels = [
            f"{pred['label']} ({pred['score']:.2f})"
            for pred in predictions
            if pred['score'] > LABEL_THRESHOLDS.get(pred['label'], 0.5)
        ]

        if triggered_labels:
            return f"⚠️ Detected: {', '.join(triggered_labels)}"
        return "✅ No hate speech detected."

    except Exception as e:
        return f"❌ Error during detection: {str(e)}"