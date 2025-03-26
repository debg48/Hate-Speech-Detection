import pinecone
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

def init_pinecone():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    return pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

def query_pinecone(embedding: List[float], top_k: int = 5) -> Dict:
    """Query Pinecone for similar embeddings"""
    index = init_pinecone()
    return index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

def upsert_embeddings(embeddings: List[Dict]):
    """Insert embeddings into Pinecone"""
    index = init_pinecone()
    return index.upsert(vectors=embeddings)