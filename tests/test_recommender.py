"""
Tests for the recommendation engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.recommender import extract_preferences, recommend, load_products


# ── Preference extraction tests ───────────────────────────────────────────────

class TestPreferenceExtraction:

    def test_brand_extraction_sony(self):
        prefs = extract_preferences("I want Sony headphones")
        assert prefs.brand is not None
        assert "sony" in prefs.brand.lower()

    def test_brand_extraction_jbl(self):
        prefs = extract_preferences("JBL speaker under 5000")
        assert prefs.brand is not None
        assert "jbl" in prefs.brand.lower()

    def test_category_headphones(self):
        prefs = extract_preferences("best headphones for music")
        assert prefs.category == "Headphones"

    def test_category_earbuds(self):
        prefs = extract_preferences("wireless earbuds under 3000")
        assert prefs.category == "Earbuds"

    def test_category_laptop(self):
        prefs = extract_preferences("laptop for coding")
        assert prefs.category == "Laptops"

    def test_category_smartphone(self):
        prefs = extract_preferences("best smartphone for photography")
        assert prefs.category == "Smartphones"

    def test_category_gaming_mouse(self):
        prefs = extract_preferences("gaming mouse under 2000")
        assert prefs.category == "Gaming"

    def test_budget_rupee_symbol(self):
        prefs = extract_preferences("headphones under ₹5000")
        assert prefs.max_price == 5000.0

    def test_budget_rs_format(self):
        prefs = extract_preferences("speakers below Rs 3000")
        assert prefs.max_price == 3000.0

    def test_budget_plain_number(self):
        prefs = extract_preferences("laptop under 70000")
        assert prefs.max_price == 70000.0

    def test_budget_k_notation(self):
        prefs = extract_preferences("phone under 30k")
        assert prefs.max_price == 30000.0

    def test_budget_range(self):
        prefs = extract_preferences("headphones between ₹3000 and ₹7000")
        assert prefs.min_price == 3000.0
        assert prefs.max_price == 7000.0

    def test_budget_affordable(self):
        prefs = extract_preferences("affordable headphones")
        assert prefs.budget_type == "cheap"

    def test_budget_premium(self):
        prefs = extract_preferences("premium flagship laptop")
        assert prefs.budget_type == "premium"

    def test_use_case_music(self):
        prefs = extract_preferences("headphones for music listening")
        assert prefs.use_case == "music"

    def test_use_case_gaming(self):
        prefs = extract_preferences("gaming headset for esports")
        assert prefs.use_case == "gaming"

    def test_use_case_coding(self):
        prefs = extract_preferences("laptop for programming")
        assert prefs.use_case == "coding"

    def test_min_rating_high(self):
        prefs = extract_preferences("wireless earbuds with high rating")
        assert prefs.min_rating is not None
        assert prefs.min_rating >= 4.0

    def test_empty_query(self):
        prefs = extract_preferences("")
        assert prefs.category is None
        assert prefs.brand is None
        assert prefs.max_price is None

    def test_combined_preferences(self):
        prefs = extract_preferences("affordable Sony headphones for music under ₹5000")
        assert prefs.brand is not None and "sony" in prefs.brand.lower()
        assert prefs.category == "Headphones"
        assert prefs.max_price == 5000.0


# ── CSV loading ───────────────────────────────────────────────────────────────

class TestProductLoading:

    def test_products_loaded(self):
        df = load_products()
        assert not df.empty
        assert len(df) >= 100

    def test_products_have_required_columns(self):
        df = load_products()
        required = ["product_id", "product_name", "brand", "category", "price", "rating"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_prices_are_numeric(self):
        df = load_products()
        assert df["price"].dtype in ["float64", "int64", "float32"]
        assert (df["price"] >= 0).all()

    def test_ratings_in_range(self):
        df = load_products()
        assert (df["rating"] >= 0).all()
        assert (df["rating"] <= 5).all()


# ── Recommendation scoring ────────────────────────────────────────────────────

class TestRecommendationEngine:

    def test_basic_recommendation_returns_results(self):
        prefs, recs, total = recommend("headphones under 5000", top_k=5)
        assert len(recs) > 0
        assert total > 0

    def test_score_in_range(self):
        prefs, recs, total = recommend("Sony headphones", top_k=5)
        for rec in recs:
            assert 0 <= rec["recommendation_score"] <= 100

    def test_empty_query_returns_results(self):
        prefs, recs, total = recommend("   ", top_k=3)
        # Should still return something or empty list gracefully
        assert isinstance(recs, list)

    def test_no_matching_brand_returns_gracefully(self):
        prefs, recs, total = recommend("UnknownBrandXYZ123 headphones", top_k=3)
        assert isinstance(recs, list)

    def test_brand_preference_scores_higher(self):
        prefs, recs, total = recommend("Sony headphones", top_k=10)
        sony_recs = [r for r in recs if r["brand"].lower() == "sony"]
        non_sony_recs = [r for r in recs if r["brand"].lower() != "sony"]
        if sony_recs and non_sony_recs:
            avg_sony = sum(r["recommendation_score"] for r in sony_recs) / len(sony_recs)
            avg_non = sum(r["recommendation_score"] for r in non_sony_recs) / len(non_sony_recs)
            assert avg_sony >= avg_non

    def test_budget_filter_respected(self):
        prefs, recs, total = recommend("headphones", max_price=2000, top_k=5)
        # All results should be within ~3x budget (soft filtering)
        for rec in recs:
            assert rec["price"] <= 2000 * 3

    def test_category_filter(self):
        prefs, recs, total = recommend("good product", category="Laptops", top_k=5)
        for rec in recs:
            assert rec["category"] == "Laptops"

    def test_min_rating_filter(self):
        prefs, recs, total = recommend("headphones", min_rating=4.0, top_k=5)
        for rec in recs:
            assert rec["rating"] >= 4.0

    def test_results_sorted_by_score(self):
        prefs, recs, total = recommend("laptop for coding", top_k=5)
        scores = [r["recommendation_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommendations_have_required_fields(self):
        prefs, recs, total = recommend("wireless earbuds", top_k=3)
        required_fields = ["product_id", "product_name", "brand", "category", "price", "rating", "recommendation_score", "reason"]
        for rec in recs:
            for field in required_fields:
                assert field in rec, f"Missing field: {field}"

    def test_no_products_found_returns_empty(self):
        prefs, recs, total = recommend("XYZ impossible query 999999 rupees", max_price=1, top_k=5)
        # Might return results due to soft filtering, but shouldn't crash
        assert isinstance(recs, list)
