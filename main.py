from fastapi import FastAPI
from models import TextInput
from fact_check import quick_fact_check
from hate_speech import detect_hate_speech

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Misinformation & Hate Speech Detection API"}

@app.post("/detect-misinformation")
async def misinformation_api(input: TextInput):
    result = await quick_fact_check(input.text)
    return {"result": result}

@app.post("/detect-hate-speech")
async def hate_speech_api(input: TextInput):
    result = detect_hate_speech(input.text)
    return {"result": result}
