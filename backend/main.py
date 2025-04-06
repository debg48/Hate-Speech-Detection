from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import TextInput
from fact_check import quick_fact_check
from hate_speech import detect_hate_speech

app = FastAPI()

# 👇 CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
