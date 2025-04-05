import httpx
import os
from dotenv import load_dotenv

load_dotenv()

FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

async def quick_fact_check(query: str) -> str:
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "languageCode": "en",
        "maxAgeDays": 180,
        "pageSize": 2,
        "key": FACT_CHECK_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(url, params=params)

        if res.status_code != 200:
            return "Fact Check API error"

        claims = res.json().get("claims", [])
        if not claims:
            return "✅ No known misinformation detected."

        claim = claims[0]
        text = claim.get("text", "No claim text.")
        review = claim.get("claimReview", [{}])[0]
        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating = review.get("textualRating", "No rating")

        return f"⚠️ Claim: {text}\nRating: {rating} (by {publisher})"

    except Exception as e:
        return f"Fact Check Error: {str(e)}"
