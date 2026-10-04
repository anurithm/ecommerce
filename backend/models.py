"""
Pydantic models for the AI E-Commerce Recommendation System.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    max_price: Optional[float] = Field(None, description="Maximum price filter in INR")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating filter")
    category: Optional[str] = Field(None, description="Category filter")
    brand: Optional[str] = Field(None, description="Brand filter")
    availability: Optional[str] = Field(None, description="Availability filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")


class ExtractedPreferences(BaseModel):
    """Preferences extracted from natural language query."""
    category: Optional[str] = None
    brand: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_rating: Optional[float] = None
    color: Optional[str] = None
    use_case: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    budget_type: Optional[str] = None  # "cheap", "premium", "specific"


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of recommendation scores."""
    semantic_score: float = 0.0
    category_score: float = 0.0
    brand_score: float = 0.0
    budget_score: float = 0.0
    rating_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0


class RecommendedProduct(BaseModel):
    """A recommended product with score and explanation."""
    product_id: str
    product_name: str
    brand: str
    category: str
    subcategory: Optional[str] = None
    price: float
    original_price: Optional[float] = None
    discount_percent: Optional[float] = None
    rating: float
    review_count: int
    description: Optional[str] = None
    features: Optional[str] = None
    color: Optional[str] = None
    availability: str
    image_url: Optional[str] = None
    recommendation_score: float
    score_breakdown: Optional[ScoreBreakdown] = None
    reason: str
    explanation_source: str = "rule-based"  # "ai" or "rule-based"


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint."""
    query: str
    preferences: ExtractedPreferences
    recommendations: List[RecommendedProduct]
    total_products_considered: int
    explanation_source: str = "rule-based"


class ProductModel(BaseModel):
    """Full product model."""
    product_id: str
    product_name: str
    brand: str
    category: str
    subcategory: Optional[str] = None
    price: float
    original_price: Optional[float] = None
    rating: float
    review_count: int
    description: Optional[str] = None
    features: Optional[str] = None
    color: Optional[str] = None
    availability: str
    image_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    embedding_model: str
    products_loaded: int
    llm_available: bool
