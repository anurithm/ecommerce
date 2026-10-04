"""
Hybrid recommendation engine with preference extraction and scoring.
"""

import re
import logging
import math
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from backend.models import ExtractedPreferences, RecommendedProduct, ScoreBreakdown
from backend.embeddings import get_product_embeddings, embed_query, cosine_similarity_single

logger = logging.getLogger(__name__)

# --- Configurable Weights ---
WEIGHTS = {
    "semantic": 0.40,
    "category": 0.20,
    "brand": 0.15,
    "budget": 0.10,
    "rating": 0.10,
    "keyword": 0.05,
}

# --- Brand / Category Aliases ---
CATEGORY_ALIASES = {
    "headphone": "Headphones",
    "headphones": "Headphones",
    "earbud": "Earbuds",
    "earbuds": "Earbuds",
    "tws": "Earbuds",
    "earphone": "Earbuds",
    "earphones": "Earbuds",
    "phone": "Smartphones",
    "phones": "Smartphones",
    "smartphone": "Smartphones",
    "smartphones": "Smartphones",
    "mobile": "Smartphones",
    "laptop": "Laptops",
    "laptops": "Laptops",
    "notebook": "Laptops",
    "smartwatch": "Smartwatches",
    "smartwatches": "Smartwatches",
    "watch": "Smartwatches",
    "fitness tracker": "Smartwatches",
    "tablet": "Tablets",
    "tablets": "Tablets",
    "camera": "Cameras",
    "cameras": "Cameras",
    "tv": "TVs",
    "television": "TVs",
    "speaker": "Speakers",
    "speakers": "Speakers",
    "gaming": "Gaming",
    "mouse": "Gaming",
    "keyboard": "Accessories",
    "webcam": "Accessories",
    "charger": "Accessories",
    "hub": "Accessories",
    "tracker": "Accessories",
    "ssd": "Accessories",
    "hdd": "Accessories",
    "storage": "Accessories",
    "streaming": "Accessories",
    "printer": "Accessories",
    "power bank": "Accessories",
}

KNOWN_BRANDS = [
    "sony", "samsung", "apple", "jbl", "boat", "sennheiser", "bose",
    "oneplus", "xiaomi", "redmi", "realme", "oppo", "vivo", "motorola",
    "asus", "dell", "hp", "lenovo", "acer", "msi", "razer", "lg",
    "garmin", "fitbit", "amazfit", "noise", "fossil", "coros",
    "logitech", "razer", "microsoft", "nintendo", "anker", "belkin",
    "western digital", "wd", "audio-technica", "skullcandy", "jabra",
    "nothing", "iqoo", "poco", "fujifilm", "gopro", "canon", "nikon",
    "zebronics", "cosmic byte", "ant esports", "fantech", "redgear",
    "marshall", "sonos", "amazon", "google",
]

BUDGET_KEYWORDS = ["cheap", "affordable", "budget", "economical", "low cost", "inexpensive", "value"]
PREMIUM_KEYWORDS = ["premium", "high-end", "luxury", "best", "top", "flagship", "professional", "pro"]

# Indian currency patterns
PRICE_PATTERNS = [
    r"₹\s*([\d,]+(?:\.\d+)?)",
    r"rs\.?\s*([\d,]+(?:\.\d+)?)",
    r"rupees?\s*([\d,]+(?:\.\d+)?)",
    r"inr\s*([\d,]+(?:\.\d+)?)",
]
PRICE_RANGE_PATTERN = r"between\s+(?:₹|rs\.?|rupees?)?\s*([\d,]+)\s+(?:and|to|-)\s+(?:₹|rs\.?|rupees?)?\s*([\d,]+)"
PLAIN_PRICE_PATTERN = r"(?:under|below|less than|within|upto|up to|max|maximum)\s+([\d,]+)\b"
AROUND_PRICE_PATTERN = r"(?:around|approximately|about|~)\s*(?:₹|rs\.?)?\s*([\d,]+)"
BETWEEN_K_PATTERN = r"([\d]+)k\b"


def _parse_price(raw: str) -> Optional[float]:
    """Parse a price string like '5,000' or '5000' into a float."""
    try:
        return float(raw.replace(",", "").strip())
    except Exception:
        return None


def extract_preferences(query: str) -> ExtractedPreferences:
    """
    Rule-based extraction of structured preferences from a natural language query.
    """
    q = query.lower().strip()
    prefs = ExtractedPreferences()

    # --- Category ---
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in q:
            prefs.category = cat
            break

    # --- Brand ---
    for brand in KNOWN_BRANDS:
        if brand in q:
            prefs.brand = brand.title()
            break

    # --- Price extraction ---
    # Range first
    m = re.search(PRICE_RANGE_PATTERN, q, re.IGNORECASE)
    if m:
        prefs.min_price = _parse_price(m.group(1))
        prefs.max_price = _parse_price(m.group(2))
    else:
        # Currency symbol patterns
        max_found = None
        for pattern in PRICE_PATTERNS:
            matches = re.findall(pattern, q, re.IGNORECASE)
            if matches:
                # Multiple prices: treat min/max
                vals = [_parse_price(p) for p in matches if _parse_price(p) is not None]
                if len(vals) == 1:
                    # Check context
                    if any(kw in q for kw in ["under", "below", "less than", "within", "max", "upto"]):
                        max_found = vals[0]
                    elif any(kw in q for kw in ["above", "more than", "at least", "min"]):
                        prefs.min_price = vals[0]
                    else:
                        max_found = vals[0]
                elif len(vals) >= 2:
                    prefs.min_price = min(vals)
                    max_found = max(vals)
                break

        if max_found is None:
            m2 = re.search(PLAIN_PRICE_PATTERN, q, re.IGNORECASE)
            if m2:
                max_found = _parse_price(m2.group(1))

        if max_found is None:
            m3 = re.search(AROUND_PRICE_PATTERN, q, re.IGNORECASE)
            if m3:
                around = _parse_price(m3.group(1))
                if around:
                    prefs.min_price = around * 0.8
                    max_found = around * 1.2

        # Handle "5k" style
        k_matches = re.findall(BETWEEN_K_PATTERN, q, re.IGNORECASE)
        if k_matches and max_found is None:
            max_found = float(k_matches[-1]) * 1000

        prefs.max_price = max_found

    # --- Budget type ---
    if any(kw in q for kw in BUDGET_KEYWORDS):
        prefs.budget_type = "cheap"
    elif any(kw in q for kw in PREMIUM_KEYWORDS):
        prefs.budget_type = "premium"
    elif prefs.max_price:
        prefs.budget_type = "specific"

    # --- Minimum rating ---
    m_rating = re.search(r"(?:rating|rated|star)\s*(?:above|over|at least|≥|>=?)?\s*([0-9]\.?[0-9]?)\s*", q)
    if m_rating:
        try:
            prefs.min_rating = float(m_rating.group(1))
        except Exception:
            pass
    if "high rating" in q or "highly rated" in q or "best rated" in q:
        prefs.min_rating = prefs.min_rating or 4.0

    # --- Color ---
    colors = ["black", "white", "blue", "red", "green", "silver", "gold", "grey", "gray", "rose", "pink"]
    for color in colors:
        if color in q:
            prefs.color = color.title()
            break

    # --- Keywords ---
    stop_words = {
        "i", "want", "need", "looking", "for", "a", "an", "the", "good", "me", "my",
        "buy", "get", "find", "show", "please", "can", "you", "some", "under",
        "below", "less", "than", "around", "within", "with", "and", "or", "is",
        "are", "best", "any", "most", "more", "very",
    }
    words = re.findall(r"\b[a-z]+\b", q)
    prefs.keywords = [w for w in words if len(w) > 3 and w not in stop_words][:10]

    # --- Use case ---
    use_cases = {
        "music": ["music", "listening", "audio", "song", "songs"],
        "gaming": ["gaming", "game", "gamer", "esports"],
        "coding": ["coding", "programming", "developer", "development", "code"],
        "photography": ["photography", "photo", "camera", "selfie", "picture"],
        "video": ["video", "vlog", "youtube", "streaming", "editing"],
        "calls": ["calls", "calling", "meetings", "conference", "work from home", "wfh"],
        "travel": ["travel", "commute", "outdoor", "portable", "on the go"],
        "gym": ["gym", "workout", "exercise", "fitness", "running", "sports"],
        "student": ["student", "college", "university", "school", "study"],
        "business": ["business", "office", "professional", "work"],
    }
    for use, kws in use_cases.items():
        if any(kw in q for kw in kws):
            prefs.use_case = use
            if use not in prefs.keywords:
                prefs.keywords.append(use)
            break

    return prefs


# ---- Scoring ----

def _score_brand(product_brand: str, pref_brand: Optional[str]) -> float:
    if pref_brand is None:
        return 0.5  # neutral
    if product_brand.lower() == pref_brand.lower():
        return 1.0
    return 0.0


def _score_category(product_category: str, pref_category: Optional[str]) -> float:
    if pref_category is None:
        return 0.5
    if product_category.lower() == pref_category.lower():
        return 1.0
    return 0.0


def _score_budget(
    price: float,
    max_price: Optional[float],
    min_price: Optional[float],
    budget_type: Optional[str],
) -> float:
    if max_price is None and min_price is None and budget_type is None:
        return 0.5  # neutral

    if max_price is not None:
        if price <= max_price:
            # How far below budget? Better score for closer to budget (sweet spot)
            ratio = price / max_price
            if ratio <= 0.5:
                return 0.7  # Very cheap - may not be right quality
            elif ratio <= 0.8:
                return 0.9
            else:
                return 1.0
        else:
            # Over budget - penalise proportionally
            over_ratio = (price - max_price) / max_price
            return max(0.0, 1.0 - over_ratio * 2)

    if min_price is not None and price < min_price:
        return 0.5

    if budget_type == "cheap":
        # No explicit price. Prefer cheaper products
        if price < 5000:
            return 1.0
        elif price < 15000:
            return 0.7
        elif price < 30000:
            return 0.4
        else:
            return 0.2

    if budget_type == "premium":
        if price > 50000:
            return 1.0
        elif price > 20000:
            return 0.7
        elif price > 10000:
            return 0.5
        else:
            return 0.3

    return 0.5


def _score_rating(rating: float, min_rating: Optional[float]) -> float:
    if min_rating is not None:
        if rating < min_rating:
            return 0.0
        return rating / 5.0
    # Normalise rating to 0-1
    return rating / 5.0


def _score_keywords(product_text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.5
    pt_lower = product_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in pt_lower)
    return min(1.0, matches / max(len(keywords), 1))


def compute_hybrid_score(
    semantic_sim: float,
    product: pd.Series,
    prefs: ExtractedPreferences,
    product_text: str,
) -> Tuple[float, ScoreBreakdown]:
    """Compute the hybrid recommendation score."""

    semantic_score = float(semantic_sim)
    category_score = _score_category(str(product.get("category", "")), prefs.category)
    brand_score = _score_brand(str(product.get("brand", "")), prefs.brand)
    budget_score = _score_budget(
        float(product.get("price", 0)),
        prefs.max_price,
        prefs.min_price,
        prefs.budget_type,
    )
    rating_score = _score_rating(float(product.get("rating", 0)), prefs.min_rating)
    keyword_score = _score_keywords(product_text, prefs.keywords)

    final = (
        WEIGHTS["semantic"] * semantic_score
        + WEIGHTS["category"] * category_score
        + WEIGHTS["brand"] * brand_score
        + WEIGHTS["budget"] * budget_score
        + WEIGHTS["rating"] * rating_score
        + WEIGHTS["keyword"] * keyword_score
    )

    breakdown = ScoreBreakdown(
        semantic_score=round(semantic_score * 100, 1),
        category_score=round(category_score * 100, 1),
        brand_score=round(brand_score * 100, 1),
        budget_score=round(budget_score * 100, 1),
        rating_score=round(rating_score * 100, 1),
        keyword_score=round(keyword_score * 100, 1),
        final_score=round(final * 100, 1),
    )

    return round(final * 100, 1), breakdown


def build_rule_based_explanation(
    product: pd.Series,
    prefs: ExtractedPreferences,
    score: float,
    breakdown: ScoreBreakdown,
) -> str:
    """Build a deterministic human-readable explanation."""
    reasons = []

    if prefs.brand and str(product.get("brand", "")).lower() == prefs.brand.lower():
        reasons.append(f"matches your {prefs.brand} brand preference")

    if prefs.category and str(product.get("category", "")).lower() == prefs.category.lower():
        reasons.append(f"is in the {prefs.category} category you're looking for")

    price = float(product.get("price", 0))
    if prefs.max_price and price <= prefs.max_price:
        reasons.append(f"fits within your ₹{prefs.max_price:,.0f} budget at ₹{price:,.0f}")
    elif prefs.budget_type == "cheap" and price < 10000:
        reasons.append("is budget-friendly")

    rating = float(product.get("rating", 0))
    if rating >= 4.4:
        reasons.append(f"is highly rated at {rating}★")
    elif rating >= 4.0:
        reasons.append(f"has good customer reviews ({rating}★)")

    if prefs.use_case:
        reasons.append(f"is suitable for {prefs.use_case}")

    if prefs.keywords:
        matching = [kw for kw in prefs.keywords if kw.lower() in str(product.get("description", "")).lower() or kw.lower() in str(product.get("features", "")).lower()]
        if matching:
            reasons.append(f"features align with your needs ({', '.join(matching[:3])})")

    if not reasons:
        reasons.append(f"scored {score:.0f}% match for your query")

    explanation = "Recommended because it " + " and ".join(reasons[:3]) + "."
    return explanation.capitalize()


# ---- Main recommend function ----

_products_df: Optional[pd.DataFrame] = None
_product_embeddings: Optional[np.ndarray] = None
_product_texts: Optional[List[str]] = None
_embedding_model_type: str = "unknown"


def load_products(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load and cache products from CSV."""
    global _products_df

    if _products_df is not None and not _products_df.empty:
        return _products_df

    if csv_path is None:
        csv_path = str(Path(__file__).parent.parent / "data" / "products.csv")

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        # Clean up columns
        df.columns = df.columns.str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["product_id", "product_name"])
        df = df.reset_index(drop=True)
        if df.empty:
            logger.error(f"CSV loaded but no valid rows from {csv_path}")
        else:
            _products_df = df
            logger.info(f"Loaded {len(df)} products from {csv_path}")
    except FileNotFoundError:
        logger.error(f"Products CSV not found at {csv_path}")
        if _products_df is None:
            _products_df = pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        if _products_df is None:
            _products_df = pd.DataFrame()

    return _products_df if _products_df is not None else pd.DataFrame()


def _ensure_embeddings(df: pd.DataFrame) -> None:
    """Ensure product embeddings are loaded."""
    global _product_embeddings, _product_texts, _embedding_model_type

    if _product_embeddings is not None and len(_product_embeddings) == len(df):
        return

    from backend.embeddings import create_product_text

    _product_texts = df.apply(create_product_text, axis=1).tolist()
    _product_embeddings, _embedding_model_type = get_product_embeddings(df)
    logger.info(f"Product embeddings ready ({_embedding_model_type}): {_product_embeddings.shape}")


def recommend(
    query: str,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    availability: Optional[str] = None,
    top_k: int = 5,
    recent_queries: Optional[List[str]] = None,
) -> Tuple[ExtractedPreferences, List[Dict[str, Any]], int]:
    """
    Main recommendation function.
    Returns (preferences, recommendations, total_considered)
    """
    df = load_products()
    if df.empty:
        return ExtractedPreferences(), [], 0

    # Extract preferences from natural language
    prefs = extract_preferences(query)

    # Override with explicit filter params if provided
    if max_price is not None:
        prefs.max_price = max_price
    if min_rating is not None:
        prefs.min_rating = min_rating
    if category is not None:
        prefs.category = category.title() if category else None
    if brand is not None:
        prefs.brand = brand.title() if brand else None

    # Personalization: blend recent query context into prefs keywords
    if recent_queries:
        recent_text = " ".join(recent_queries[-3:])
        recent_prefs = extract_preferences(recent_text)
        if not prefs.category and recent_prefs.category:
            prefs.keywords.extend([recent_prefs.category.lower()])
        if recent_prefs.keywords:
            prefs.keywords.extend(recent_prefs.keywords[:3])
        prefs.keywords = list(dict.fromkeys(prefs.keywords))[:15]

    # Apply hard filters
    filtered_df = df.copy()

    if availability and availability.lower() != "all":
        filtered_df = filtered_df[
            filtered_df["availability"].str.lower() == availability.lower()
        ]

    if prefs.min_rating is not None:
        filtered_df = filtered_df[filtered_df["rating"] >= prefs.min_rating]

    # Soft filter on price (don't exclude everything, just penalise in score)
    if prefs.max_price is not None:
        # Hard-exclude only extremely over-budget items (> 3x budget)
        filtered_df = filtered_df[filtered_df["price"] <= prefs.max_price * 3]

    if filtered_df.empty:
        # Fall back to full df
        filtered_df = df.copy()

    total_considered = len(filtered_df)

    # Ensure embeddings exist
    _ensure_embeddings(df)

    # Get indices in original df for filtered rows
    filtered_indices = filtered_df.index.tolist()

    if _product_embeddings is not None and len(_product_embeddings) > 0:
        query_vec = embed_query(query)
        product_vecs = _product_embeddings[filtered_indices]
        similarities = cosine_similarity_single(query_vec, product_vecs)
    else:
        similarities = np.zeros(len(filtered_df))

    # Score all filtered products
    results = []
    for i, (orig_idx, row) in enumerate(filtered_df.iterrows()):
        sem_sim = float(similarities[i]) if i < len(similarities) else 0.0
        pt = _product_texts[orig_idx] if _product_texts and orig_idx < len(_product_texts) else ""
        final_score, breakdown = compute_hybrid_score(sem_sim, row, prefs, pt)
        if final_score > 20.0:  # Minimum relevancy threshold
            results.append((final_score, breakdown, row))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    recommendations = []
    for score, breakdown, row in top_results:
        rule_reason = build_rule_based_explanation(row, prefs, score, breakdown)

        orig_price = float(row["original_price"]) if pd.notna(row.get("original_price")) else None
        price = float(row["price"])
        discount = None
        if orig_price and orig_price > price:
            discount = round((orig_price - price) / orig_price * 100, 1)

        rec = {
            "product_id": str(row["product_id"]),
            "product_name": str(row["product_name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "subcategory": str(row.get("subcategory", "")) if pd.notna(row.get("subcategory")) else None,
            "price": price,
            "original_price": orig_price,
            "discount_percent": discount,
            "rating": float(row["rating"]),
            "review_count": int(row["review_count"]),
            "description": str(row["description"]) if pd.notna(row.get("description")) else None,
            "features": str(row["features"]) if pd.notna(row.get("features")) else None,
            "color": str(row["color"]) if pd.notna(row.get("color")) else None,
            "availability": str(row["availability"]),
            "image_url": str(row["image_url"]) if pd.notna(row.get("image_url")) else None,
            "recommendation_score": score,
            "score_breakdown": breakdown,
            "reason": rule_reason,
            "explanation_source": "rule-based",
        }
        recommendations.append(rec)

    return prefs, recommendations, total_considered
