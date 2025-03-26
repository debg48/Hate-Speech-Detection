from vertexai.language_models import TextEmbeddingModel
import vertexai
import os
from dotenv import load_dotenv
from typing import List
import concurrent.futures

load_dotenv()

vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION"),
)

def generate_embedding_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Generate embeddings in parallel batches"""
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    def process_batch(batch):
        embeddings = model.get_embeddings(batch)
        return [e.values for e in embeddings]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    
    return results