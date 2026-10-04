"""
LLM integration with OpenRouter API and automatic fallback.
"""

import logging
import os
from typing import Optional, Dict, Any

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

LLM_AVAILABLE = bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_KEY_HERE")


def is_llm_available() -> bool:
    """Check if LLM is configured."""
    return bool(
        os.getenv("OPENROUTER_API_KEY", "")
        and os.getenv("OPENROUTER_API_KEY", "") != "YOUR_KEY_HERE"
    )


async def generate_recommendation_explanation(
    user_query: str,
    product: Dict[str, Any],
    score: float,
    extracted_preferences: Dict[str, Any],
) -> tuple[str, str]:
    """
    Generate an AI explanation for why a product is recommended.

    Returns:
        (explanation_text, source) where source is "ai" or "rule-based"
    """
    if not is_llm_available():
        return _fallback_explanation(product, score, extracted_preferences), "rule-based"

    prompt = _build_prompt(user_query, product, score, extracted_preferences)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "AI E-Commerce Recommender",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.4,
                },
            )

        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            # Ensure it's concise
            text = text.split("\n")[0][:400]
            if text:
                return text, "ai"
        else:
            logger.warning(f"OpenRouter returned {response.status_code}: {response.text[:200]}")

    except httpx.TimeoutException:
        logger.warning("OpenRouter request timed out, using fallback")
    except Exception as e:
        logger.warning(f"OpenRouter request failed: {e}, using fallback")

    return _fallback_explanation(product, score, extracted_preferences), "rule-based"


def _build_prompt(
    user_query: str,
    product: Dict[str, Any],
    score: float,
    prefs: Dict[str, Any],
) -> str:
    """Build the LLM prompt."""
    pref_parts = []
    if prefs.get("brand"):
        pref_parts.append(f"brand preference: {prefs['brand']}")
    if prefs.get("max_price"):
        pref_parts.append(f"budget: ₹{prefs['max_price']:,.0f}")
    if prefs.get("category"):
        pref_parts.append(f"category: {prefs['category']}")
    if prefs.get("use_case"):
        pref_parts.append(f"use case: {prefs['use_case']}")

    pref_str = ", ".join(pref_parts) if pref_parts else "no specific preferences"

    return f"""You are an e-commerce product recommendation assistant. Explain in ONE concise sentence (max 60 words) why this product is recommended for the customer. Be specific and do not invent features not listed.

Customer query: "{user_query}"
Customer preferences: {pref_str}

Product: {product.get('product_name')} by {product.get('brand')}
Price: ₹{product.get('price', 0):,.0f}
Rating: {product.get('rating')}★
Category: {product.get('category')}
Features: {product.get('features', 'N/A')}
Match score: {score:.0f}%

Write one sentence starting with "Recommended because"."""


def _fallback_explanation(
    product: Dict[str, Any],
    score: float,
    prefs: Dict[str, Any],
) -> str:
    """Generate a deterministic rule-based fallback explanation."""
    reasons = []

    if prefs.get("brand") and str(product.get("brand", "")).lower() == str(prefs.get("brand", "")).lower():
        reasons.append(f"matches your {prefs['brand']} brand preference")

    price = product.get("price", 0)
    max_price = prefs.get("max_price")
    if max_price and price <= max_price:
        reasons.append(f"fits within your ₹{max_price:,.0f} budget")
    elif prefs.get("budget_type") == "cheap":
        reasons.append("is an affordable option")

    rating = product.get("rating", 0)
    if rating >= 4.5:
        reasons.append(f"is highly rated at {rating}★")
    elif rating >= 4.0:
        reasons.append(f"has solid customer reviews ({rating}★)")

    if prefs.get("category") and str(product.get("category", "")).lower() == str(prefs.get("category", "")).lower():
        reasons.append(f"is in the {prefs['category']} category you're looking for")

    if not reasons:
        reasons = [
            "matches your selected category and budget",
            "has strong customer ratings",
        ]

    explanation = "Recommended because it " + " and ".join(reasons[:3]) + "."
    return explanation.capitalize()


async def generate_bulk_explanations(
    user_query: str,
    recommendations: list,
    extracted_preferences: Dict[str, Any],
) -> list:
    """
    Generate explanations for all recommendations.
    Uses AI for top recommendations when available.
    """
    import asyncio

    tasks = []
    for rec in recommendations:
        task = generate_recommendation_explanation(
            user_query=user_query,
            product=rec,
            score=rec.get("recommendation_score", 0),
            extracted_preferences=extracted_preferences,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    updated = []
    for rec, result in zip(recommendations, results):
        if isinstance(result, Exception):
            rec = dict(rec)
            rec["reason"] = _fallback_explanation(rec, rec.get("recommendation_score", 0), extracted_preferences)
            rec["explanation_source"] = "rule-based"
        else:
            explanation, source = result
            rec = dict(rec)
            rec["reason"] = explanation
            rec["explanation_source"] = source
        updated.append(rec)

    return updated
