import os
import json
from dotenv import load_dotenv
from fastapi import HTTPException
from googleapiclient import discovery

load_dotenv()

API_KEY = os.getenv("PERSPECTIVE_API_KEY")  # Set this in your .env file

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

THRESHOLDS = {
    "TOXICITY": 0.5,
    "INSULT": 0.5,
    "THREAT": 0.5,
    "OBSCENE": 0.5,
    "IDENTITY_ATTACK": 0.4,
    "SEVERE_TOXICITY": 0.3
}

def detect_hate_speech(text: str) -> dict:
    try:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in THRESHOLDS.keys()}
        }

        response = client.comments().analyze(body=analyze_request).execute()

        detected_labels = []
        for attr, data in response.get("attributeScores", {}).items():
            score = data["summaryScore"]["value"]
            if score >= THRESHOLDS.get(attr, 0.5):
                detected_labels.append({"label": attr, "score": round(score, 2)})

        return {
            "hate_speech": bool(detected_labels),
            "detected_labels": detected_labels,
            "message": "⚠️ Detected hate speech." if detected_labels else "✅ No hate speech detected."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
