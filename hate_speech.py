import os
import requests
from dotenv import load_dotenv
from fastapi import HTTPException

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

def detect_hate_speech(text: str) -> dict:
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=10
        )
        response.raise_for_status()

        result = response.json()

        # If Hugging Face returns an error inside a 200 response
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=502, detail=f"HuggingFace Error: {result['error']}")

        predictions = result[0]

        triggered_labels = [
            {"label": pred["label"], "score": round(pred["score"], 2)}
            for pred in predictions
            if pred["score"] > LABEL_THRESHOLDS.get(pred["label"], 0.5)
        ]

        return {
            "hate_speech": bool(triggered_labels),
            "detected_labels": triggered_labels,
            "message": "⚠️ Detected hate speech." if triggered_labels else "✅ No hate speech detected."
        }

    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=f"HTTP Error: {http_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
