import httpx
import os
from dotenv import load_dotenv

load_dotenv()

FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")

async def quick_fact_check(query: str) -> dict:
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
            return {
                "misinformation": False,
                "message": "❌ Fact Check API error",
                "details": None
            }

        claims = res.json().get("claims", [])
        if not claims:
            return {
                "claims": False,
                "message": "No known claims",
                "details": None
            }

        claim = claims[0]
        text = claim.get("text", "No claim text.")
        review = claim.get("claimReview", [{}])[0]
        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating = review.get("textualRating", "No rating")

        return {
            "claims" : True,
            "message": "⚠️ Claims found.",
            "details": {
                "claim": text,
                "rating": rating,
                "publisher": publisher
            }
        }

    except Exception as e:
        return {
            "misinformation": False,
            "message": f"❌ Fact Check Error: {str(e)}",
            "details": None
        }