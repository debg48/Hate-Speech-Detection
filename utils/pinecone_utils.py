import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Union

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get the index host from env
INDEX_HOST = os.getenv("PINECONE_INDEX_NAME")

if not INDEX_HOST:
    raise ValueError("PINECONE_INDEX_NAME is not set in environment variables.")

# Connect to the index
index = pc.Index(host=INDEX_HOST)

def query_pinecone(query_text: str, top_k: int = 5) -> Union[Dict, None]:
    """Query Pinecone for similar records"""
    if not query_text:
        raise ValueError("Query text cannot be empty.")

    try:
        response = index.search_records(
            namespace="default-namespace",
            query={"inputs": {"text": query_text}, "top_k": top_k},
            fields=["category", "chunk_text"]
        )
        return response
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

def upsert_embeddings(embeddings: List[Dict[str, Union[str, List[float], Dict]]]) -> Union[Dict, None]:
    """Insert text records into Pinecone"""
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty.")

    formatted_records = []
    for entry in embeddings:
        # Ensure expected keys exist
        if "id" not in entry or "vector" not in entry:
            print(f"Skipping invalid entry (missing 'id' or 'vector'): {entry}")
            continue
        
        formatted_records.append({
            "id": entry["id"],  # Make sure this exists in the input
            "values": entry["vector"],  # Pinecone expects "values"
            "metadata": entry.get("metadata", {})  # Optional metadata
        })

    if not formatted_records:
        print("Warning: No valid records to upsert.")
        return None

    try:
        print(f"Upserting {len(formatted_records)} records...")
        response = index.upsert_records(
            namespace="default-namespace",
            records=formatted_records
        )
        print(f"Upsert Response: {response}")
        return response
    except Exception as e:
        print(f"Error upserting records: {e}")
        return None