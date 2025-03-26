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
import requests
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

@app.post("/update-embeddings", summary="Update embeddings from public GCS CSV")
async def update_embeddings():
    """Process public CSV from GCS and update Pinecone embeddings"""
    temp_file = None
    try:
        # Fixed GCS URI for public file
        GCS_URI = "https://storage.googleapis.com/hate-speech-data-1743017900/filtered100.csv"
        logger.info(f"Processing CSV from GCS: {GCS_URI}")
        
        # Download CSV from public GCS URL
        try:
            response = requests.get(GCS_URI)
            response.raise_for_status()
            
            # Create temp file
            temp_file = NamedTemporaryFile(suffix=".csv", delete=False)
            temp_file.write(response.content)
            temp_file.close()  # Close to allow reading
            
            # Process CSV
            df = pd.read_csv(temp_file.name)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download CSV: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
        
        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        if 'content' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain 'Content' column. Found columns: {list(df.columns)}"
            )
        
        # Generate embeddings in batches
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
                "index": idx  # Adding index for reference
            }
        } for idx, (text, embedding) in enumerate(zip(texts, embeddings))]
        
        # Upsert to Pinecone
        response = upsert_embeddings(vectors)
        logger.info(f"Upserted {response.upserted_count} vectors from batch {batch_id}")
        
        return {
            "status": "success",
            "upserted_count": response.upserted_count,
            "total_processed": len(vectors),
            "batch_id": batch_id,
            "sample_text": texts[0] if texts else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")

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