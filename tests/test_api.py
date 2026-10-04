"""
API integration tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db

# Initialize database for tests
init_db()

client = TestClient(app)


class TestAPIHealth:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "products_loaded" in data
        assert data["products_loaded"] > 0

    def test_categories_endpoint(self):
        response = client.get("/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "brands" in data
        assert len(data["categories"]) > 0

    def test_products_endpoint(self):
        response = client.get("/products")
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert "total" in data
        assert data["total"] >= 100

    def test_products_with_category_filter(self):
        response = client.get("/products", params={"category": "Headphones"})
        assert response.status_code == 200
        data = response.json()
        for p in data["products"]:
            assert p["category"].lower() == "headphones"

    def test_single_product_endpoint(self):
        response = client.get("/products/P001")
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "P001"

    def test_product_not_found(self):
        response = client.get("/products/NONEXISTENT999")
        assert response.status_code == 404


class TestRecommendationAPI:

    def test_recommend_basic(self):
        response = client.post("/recommend", json={"query": "headphones"})
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "preferences" in data
        assert "query" in data

    def test_recommend_with_brand(self):
        response = client.post("/recommend", json={
            "query": "Sony headphones under 5000"
        })
        assert response.status_code == 200
        data = response.json()
        prefs = data["preferences"]
        assert prefs.get("max_price") == 5000.0

    def test_recommend_returns_scores(self):
        response = client.post("/recommend", json={"query": "wireless earbuds"})
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        for rec in recs:
            assert "recommendation_score" in rec
            assert 0 <= rec["recommendation_score"] <= 100

    def test_recommend_returns_reason(self):
        response = client.post("/recommend", json={"query": "gaming mouse"})
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        for rec in recs:
            assert "reason" in rec
            assert len(rec["reason"]) > 0

    def test_recommend_empty_query_rejected(self):
        response = client.post("/recommend", json={"query": ""})
        assert response.status_code == 422 or response.status_code == 400

    def test_recommend_laptop_coding(self):
        response = client.post("/recommend", json={
            "query": "best laptop for coding under 70000"
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) > 0

    def test_recommend_with_filters(self):
        response = client.post("/recommend", json={
            "query": "smartphone",
            "max_price": 30000,
            "min_rating": 4.0,
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        for rec in data["recommendations"]:
            assert rec["rating"] >= 4.0

    def test_recommend_history_stored(self):
        # Make a recommendation
        client.post("/recommend", json={"query": "test history query unique"})
        # Check history
        response = client.get("/history", params={"limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
