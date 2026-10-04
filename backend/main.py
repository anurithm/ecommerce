"""
FastAPI backend for the AI E-Commerce Recommendation System.
"""

import logging
import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.models import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
    ExtractedPreferences,
    HealthResponse,
    ProductModel,
    ScoreBreakdown,
)
from backend.recommender import load_products, recommend, extract_preferences
from backend.embeddings import get_model_type, get_product_embeddings
from backend.llm import is_llm_available, generate_bulk_explanations
from backend import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI E-Commerce Recommendation API",
    description="AI-powered product recommendation system using hybrid scoring and LLM explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load products and embeddings on startup."""
    logger.info("Starting AI E-Commerce Recommendation API...")
    database.init_db()
    df = load_products()
    if not df.empty:
        try:
            get_product_embeddings(df)
        except Exception as e:
            logger.warning(f"Could not pre-load embeddings: {e}")
    logger.info(f"Startup complete. Products loaded: {len(df)}, LLM available: {is_llm_available()}")


@app.get("/", tags=["Status"])
async def root():
    return {"status": "ok", "message": "AI E-Commerce Recommendation API"}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    df = load_products()
    return HealthResponse(
        status="ok",
        message="Service is running",
        embedding_model=get_model_type(),
        products_loaded=len(df),
        llm_available=is_llm_available(),
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get AI-powered product recommendations for a natural language query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Get recent search context for personalization
        recent_queries = database.get_recent_preferences(limit=5)

        prefs, recommendations, total_considered = recommend(
            query=request.query,
            max_price=request.max_price,
            min_rating=request.min_rating,
            category=request.category,
            brand=request.brand,
            availability=request.availability,
            top_k=request.top_k,
            recent_queries=recent_queries,
        )

        if not recommendations:
            return RecommendationResponse(
                query=request.query,
                preferences=prefs,
                recommendations=[],
                total_products_considered=total_considered,
                explanation_source="rule-based",
            )

        # Generate AI explanations (with fallback)
        prefs_dict = prefs.model_dump()
        recommendations = await generate_bulk_explanations(
            user_query=request.query,
            recommendations=recommendations,
            extracted_preferences=prefs_dict,
        )

        # Determine overall explanation source
        sources = [r.get("explanation_source", "rule-based") for r in recommendations]
        overall_source = "ai" if "ai" in sources else "rule-based"

        # Save to history
        product_ids = [r["product_id"] for r in recommendations]
        database.save_recommendation(
            query=request.query,
            preferences=prefs_dict,
            product_ids=product_ids,
        )

        # Convert to RecommendedProduct models
        rec_models = []
        for r in recommendations:
            breakdown = r.get("score_breakdown")
            if isinstance(breakdown, dict):
                breakdown = ScoreBreakdown(**breakdown)
            rec_models.append(
                RecommendedProduct(
                    product_id=r["product_id"],
                    product_name=r["product_name"],
                    brand=r["brand"],
                    category=r["category"],
                    subcategory=r.get("subcategory"),
                    price=r["price"],
                    original_price=r.get("original_price"),
                    discount_percent=r.get("discount_percent"),
                    rating=r["rating"],
                    review_count=r["review_count"],
                    description=r.get("description"),
                    features=r.get("features"),
                    color=r.get("color"),
                    availability=r["availability"],
                    image_url=r.get("image_url"),
                    recommendation_score=r["recommendation_score"],
                    score_breakdown=breakdown,
                    reason=r["reason"],
                    explanation_source=r.get("explanation_source", "rule-based"),
                )
            )

        return RecommendationResponse(
            query=request.query,
            preferences=prefs,
            recommendations=rec_models,
            total_products_considered=total_considered,
            explanation_source=overall_source,
        )

    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/products", tags=["Products"])
async def get_products(
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get all products with optional filters."""
    df = load_products()
    if df.empty:
        return {"products": [], "total": 0}

    filtered = df.copy()
    if category:
        filtered = filtered[filtered["category"].str.lower() == category.lower()]
    if brand:
        filtered = filtered[filtered["brand"].str.lower() == brand.lower()]
    if min_price is not None:
        filtered = filtered[filtered["price"] >= min_price]
    if max_price is not None:
        filtered = filtered[filtered["price"] <= max_price]
    if min_rating is not None:
        filtered = filtered[filtered["rating"] >= min_rating]

    total = len(filtered)
    page = filtered.iloc[offset : offset + limit]
    products = page.where(page.notna(), None).to_dict(orient="records")

    return {"products": products, "total": total, "limit": limit, "offset": offset}


@app.get("/products/{product_id}", tags=["Products"])
async def get_product(product_id: str):
    """Get a single product by ID."""
    df = load_products()
    result = df[df["product_id"] == product_id]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    product = result.iloc[0].where(result.iloc[0].notna(), None).to_dict()
    return product


@app.get("/categories", tags=["Products"])
async def get_categories():
    """Get all unique categories and brands."""
    df = load_products()
    return {
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "brands": sorted(df["brand"].dropna().unique().tolist()),
    }


@app.get("/history", tags=["History"])
async def get_history(limit: int = Query(5, ge=1, le=20)):
    """Get recent recommendation history."""
    history = database.get_recent_history(limit=limit)
    return {"history": history, "count": len(history)}
