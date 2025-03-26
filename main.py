from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyHttpUrl
import pandas as pd
from utils.embeddings import generate_embedding_batch
from utils.pinecone_utils import upsert_embeddings, query_pinecone
from utils.gcs_utils import download_csv_from_gcs
import os
from dotenv import load_dotenv
from typing import List
import uuid
import logging
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Hate Speech Detection API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class GCSRequest(BaseModel):
    gcs_uri: str  # Format: "gs://bucket-name/path/to/file.csv"

@app.post("/update-embeddings-from-gcs", summary="Update embeddings from GCS CSV")
async def update_embeddings_from_gcs(request: GCSRequest):
    """Process CSV from GCS and update Pinecone embeddings"""
    temp_file = None
    try:
        # Validate GCS URI format
        if not request.gcs_uri.startswith("gs://"):
            raise HTTPException(status_code=400, detail="Invalid GCS URI format. Must start with gs://")
        
        logger.info(f"Processing CSV from GCS: {request.gcs_uri}")
        
        # Download CSV from GCS to temp file
        temp_file = download_csv_from_gcs(request.gcs_uri)
        
        # Process CSV
        try:
            df = pd.read_csv(temp_file.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
        
        if 'content' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'content' column")
        
        # Generate embeddings in batches
        texts = df['content'].tolist()
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = generate_embedding_batch(texts)
        
        # Prepare vectors for Pinecone
        vectors = [{
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": text,
                "source": request.gcs_uri,
                "batch_id": str(uuid.uuid4())  # For tracking
            }
        } for text, embedding in zip(texts, embeddings)]
        
        # Upsert to Pinecone
        response = upsert_embeddings(vectors)
        logger.info(f"Upserted {response.upserted_count} vectors")
        
        return {
            "status": "success",
            "upserted_count": response.upserted_count,
            "total_processed": len(vectors),
            "batch_id": vectors[0]["metadata"]["batch_id"] if vectors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_file:
            temp_file.close()
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/detect-hate-speech", summary="Detect hate speech in text")
async def detect_hate_speech(request: TextRequest):
    """Check if input text matches known hate speech patterns"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate embedding
        embedding = generate_embedding_batch([request.text])[0]
        
        # Query Pinecone
        results = query_pinecone(embedding, top_k=5)
        
        # Process matches
        matches = [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "source": match["metadata"].get("source", "unknown")
            } 
            for match in results["matches"] 
            if match["score"] > 0.7  # Adjust threshold as needed
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