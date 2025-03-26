from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from utils.embeddings import generate_embedding_batch
from utils.pinecone_utils import upsert_embeddings, query_pinecone
import os
from dotenv import load_dotenv
from typing import List
import uuid
import logging
import requests
from tempfile import NamedTemporaryFile
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Hate Speech Detection API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get Pinecone Index
INDEX_HOST = os.getenv("PINECONE_INDEX_NAME")
if not INDEX_HOST:
    raise ValueError("PINECONE_INDEX_NAME is not set in environment variables.")

index = pc.Index(host=INDEX_HOST)

# Request Models
class TextRequest(BaseModel):
    text: str

@app.post("/update-embeddings", summary="Update embeddings from public GCS CSV")
async def update_embeddings():
    """Download a public CSV from GCS, process it, and update Pinecone embeddings."""
    temp_file = None
    try:
        GCS_URI = "https://storage.googleapis.com/hate-speech-data-1743017900/filtered100.csv"
        logger.info(f"Processing CSV from GCS: {GCS_URI}")

        # Download CSV
        try:
            response = requests.get(GCS_URI)
            response.raise_for_status()
            
            # Create a temporary file
            temp_file = NamedTemporaryFile(suffix=".csv", delete=False)
            temp_file.write(response.content)
            temp_file.close()

            # Load CSV into DataFrame
            df = pd.read_csv(temp_file.name)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download CSV: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")

        # Ensure 'content' column exists
        df.columns = df.columns.str.strip().str.lower()
        if 'content' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain 'content' column. Found: {list(df.columns)}"
            )

        # Generate embeddings
        texts = df['content'].astype(str).tolist()
        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = generate_embedding_batch(texts)

        # Prepare vectors for Pinecone
        batch_id = str(uuid.uuid4())
        vectors = [{
            "id": f"vec_{batch_id}_{idx}",
            "values": embedding,
            "metadata": {
                "text": text,
                "source": GCS_URI,
                "batch_id": batch_id,
                "index": idx
            }
        } for idx, (text, embedding) in enumerate(zip(texts, embeddings))]

        if not vectors:
            raise HTTPException(status_code=400, detail="No valid data to upsert.")

        # Upsert to Pinecone
        try:
            index.upsert(
                namespace="hate-speech-namespace",
                vectors=vectors
            )
            logger.info(f"Upserted {len(vectors)} vectors from batch {batch_id}")
            return {
                "status": "success",
                "total_upserted": len(vectors),
                "batch_id": batch_id,
                "sample_text": texts[0] if texts else None
            }
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error upserting data to Pinecone")

    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")

@app.post("/detect-hate-speech", summary="Detect hate speech in text")
async def detect_hate_speech(request: TextRequest):
    """Analyze text and detect if it contains hate speech."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Generate embedding for input text
        embedding = generate_embedding_batch([request.text])[0]

        # Query Pinecone
        results = query_pinecone(embedding, top_k=5)

        if not results or "matches" not in results:
            logger.error("Pinecone query returned no matches or an invalid response")
            return {
                "is_hate_speech": False,
                "matches": [],
                "match_count": 0
            }

        # Extract matches
        matches = [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "source": match["metadata"].get("source", "unknown")
            }
            for match in results.get("matches", []) if match.get("score", 0) > 0.6
        ]

        return {
            "is_hate_speech": len(matches) > 0,
            "matches": matches,
            "match_count": len(matches)
        }

    except Exception as e:
        logger.error(f"Hate speech detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
