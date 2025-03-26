from google.cloud import storage
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

def download_csv_from_gcs(gcs_uri: str) -> str:
    """Downloads a CSV from GCS to local temp file"""
    client = storage.Client()
    
    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI format")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    
    # Download to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_file.name)
        return temp_file.name
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e