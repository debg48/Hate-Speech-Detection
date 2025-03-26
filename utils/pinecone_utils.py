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

def query_pinecone(embedding, top_k=5):
    """Query Pinecone with a precomputed embedding."""
    try:
        # Ensure Pinecone index is initialized
        index = pc.Index(host=os.getenv("PINECONE_INDEX_NAME"))
        
        response = index.query(
            namespace="hate-speech-namespace",
            vector=embedding,  # Ensure you are sending an embedding, not raw text!
            top_k=top_k,
            include_metadata=True
        )

        return response

    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}", exc_info=True)
        return None  

def upsert_embeddings(vectors):
    """Upserts sparse vectors into Pinecone."""
    if not vectors:
        print("Warning: No vectors provided for upsert.")
        return None

    formatted_vectors = []
    for entry in vectors:
        if "id" not in entry or "sparse_values" not in entry:
            print(f"Skipping invalid entry (missing 'id' or 'sparse_values'): {entry}")
            continue

        formatted_vectors.append(entry)

    if not formatted_vectors:
        print("Warning: No valid vectors to upsert.")
        return None

    try:
        print(f"Upserting {len(formatted_vectors)} vectors...")
        response = index.upsert(
            namespace="example-namespace",
            vectors=formatted_vectors
        )
        print(f"Upsert Response: {response}")
        return response
    except Exception as e:
        print(f"Error upserting vectors: {e}")
        return None