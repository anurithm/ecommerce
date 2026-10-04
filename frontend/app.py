"""
AI-Powered E-Commerce Product Recommendation System
Streamlit Frontend
"""

import streamlit as st
import httpx
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI E-Commerce Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-start Backend for Streamlit Deployment ───────────────────────────────
import os
import sys
import socket
import subprocess
import time
import streamlit as st

def start_backend_if_needed():
    """Starts the FastAPI backend if it's not already running on port 8000."""
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if not is_port_in_use(8000):
        print("Backend not found on port 8000. Starting it in the background...")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        with open("backend.log", "w") as out:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT
            )
            
        with st.spinner("Starting AI Engine (this may take up to 2 minutes on first load)..."):
            start_time = time.time()
            while not is_port_in_use(8000) and (time.time() - start_time) < 120:
                time.sleep(1)
                
start_backend_if_needed()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

EXAMPLE_QUERIES = [
    "Affordable Sony headphones for music under ₹5000",
    "Best laptop for coding under ₹70000",
    "Gaming mouse under ₹2000",
    "Good smartphone for photography",
    "Wireless earbuds with high rating",
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero section */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* Search bar */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}

/* Product card */
.product-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.product-card:hover {
    box-shadow: 0 8px 24px rgba(99,102,241,0.12);
    transform: translateY(-2px);
}
.product-card-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.product-image {
    width: 90px;
    height: 70px;
    object-fit: cover;
    border-radius: 10px;
    background: #f1f5f9;
    flex-shrink: 0;
}
.product-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.15rem;
    line-height: 1.3;
}
.product-brand-cat {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.price-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
    flex-wrap: wrap;
}
.price-main {
    font-size: 1.3rem;
    font-weight: 700;
    color: #059669;
}
.price-original {
    font-size: 0.9rem;
    color: #94a3b8;
    text-decoration: line-through;
}
.price-discount {
    font-size: 0.78rem;
    font-weight: 600;
    color: #dc2626;
    background: #fee2e2;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
}
.rating-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: #334155;
}
.rating-star {
    color: #f59e0b;
    font-size: 0.95rem;
}

/* Score badge */
.score-badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    text-align: center;
}
.score-number {
    font-size: 1.1rem;
    display: block;
    line-height: 1;
}

/* Explanation box */
.explanation-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.75rem;
    font-size: 0.88rem;
    color: #065f46;
    line-height: 1.5;
}
.explanation-ai-badge {
    display: inline-block;
    background: #10b981;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-left: 0.4rem;
    vertical-align: middle;
}
.explanation-rule-badge {
    display: inline-block;
    background: #64748b;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* Sidebar */
.sidebar-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
}

/* Example query chips */
.query-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #475569;
    margin: 0.2rem;
    cursor: pointer;
    transition: all 0.15s;
}
.query-chip:hover {
    background: #e0e7ff;
    border-color: #6366f1;
    color: #4338ca;
}

/* Tags */
.pref-tag {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
}

/* History */
.history-item {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
    color: #475569;
}

/* Availability badge */
.avail-in {
    display: inline-block;
    background: #dcfce7;
    color: #15803d;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
}
.avail-out {
    display: inline-block;
    background: #fee2e2;
    color: #b91c1c;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
}

/* Section heading */
.section-heading {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
}

/* No results */
.no-results {
    text-align: center;
    padding: 3rem 1rem;
    background: #f8fafc;
    border-radius: 16px;
    border: 2px dashed #cbd5e1;
}

/* Score bars */
.score-bar-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Make a GET request to the backend API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{BACKEND_URL}{endpoint}", params=params)
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"API error {r.status_code}: {r.text[:200]}")
                return None
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to the backend. Please make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_post(endpoint: str, data: dict) -> Optional[Dict]:
    """Make a POST request to the backend API."""
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{BACKEND_URL}{endpoint}", json=data)
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"API error {r.status_code}: {r.text[:300]}")
                return None
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def render_stars(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def format_price(price: float) -> str:
    return f"₹{price:,.0f}"


def render_product_card(rec: Dict, index: int):
    """Render a product recommendation card."""
    score = rec.get("recommendation_score", 0)
    source = rec.get("explanation_source", "rule-based")

    img_url = rec.get("image_url", "https://placehold.co/90x70/f1f5f9/94a3b8?text=Product")
    if "via.placeholder.com" in img_url:
        img_url = img_url.replace("via.placeholder.com", "placehold.co")

    with st.container():
        st.markdown(
            f"""
<div class="product-card">
    <div class="score-badge">
        <span class="score-number">{score:.0f}%</span>
        match
    </div>
    <div class="product-card-header">
        <img class="product-image" src="{img_url}"
             onerror="this.src='https://placehold.co/90x70/f1f5f9/94a3b8?text=No+Image'" />
        <div style="flex:1; min-width:0;">
            <div class="product-name">{rec.get('product_name', '')}</div>
            <div class="product-brand-cat">
                <strong>{rec.get('brand', '')}</strong> &bull; {rec.get('category', '')}
                {f" &bull; {rec.get('subcategory')}" if rec.get('subcategory') else ''}
            </div>
            <div class="price-row">
                <span class="price-main">{format_price(rec.get('price', 0))}</span>{f'''
                <span class="price-original">{format_price(rec.get("original_price"))}</span>''' if rec.get("original_price") else ""}{f'''
                <span class="price-discount">-{rec.get("discount_percent"):.0f}%</span>''' if rec.get("discount_percent") else ""}
            </div>
            <div class="rating-row">
                <span class="rating-star">{render_stars(rec.get('rating', 0))}</span>
                <span><strong>{rec.get('rating', 0)}</strong> ({rec.get('review_count', 0):,} reviews)</span>
                &nbsp;
                <span class="{'avail-in' if rec.get('availability','').lower()=='in stock' else 'avail-out'}">
                    {rec.get('availability', 'Unknown')}
                </span>
            </div>
        </div>
    </div>
    <div class="explanation-box">
        <strong>💡 Why this product?</strong>
        <span class="{'explanation-ai-badge' if source == 'ai' else 'explanation-rule-badge'}">
            {'✨ AI' if source == 'ai' else '⚙️ Rule-based'}
        </span><br/>
        {rec.get('reason', '')}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Details expander
        with st.expander(f"👆 Click here to View Product Details — {rec.get('product_name', '')[:40]}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Product Information**")
                st.write(f"**ID:** {rec.get('product_id', '')}")
                st.write(f"**Brand:** {rec.get('brand', '')}")
                st.write(f"**Category:** {rec.get('category', '')}")
                st.write(f"**Color:** {rec.get('color', 'N/A')}")
                if rec.get("description"):
                    st.markdown("**Description:**")
                    st.write(rec["description"])
                if rec.get("features"):
                    st.markdown("**Features:**")
                    features = rec["features"].strip("\"'")
                    for feat in features.split(","):
                        f = feat.strip()
                        if f:
                            st.write(f"• {f}")
            with c2:
                st.markdown("**Recommendation Breakdown**")
                breakdown = rec.get("score_breakdown", {})
                if breakdown:
                    if isinstance(breakdown, dict):
                        scores_to_show = {
                            "🔍 Semantic Match": breakdown.get("semantic_score", 0),
                            "🏷️ Category Match": breakdown.get("category_score", 0),
                            "🌟 Brand Match": breakdown.get("brand_score", 0),
                            "💰 Budget Match": breakdown.get("budget_score", 0),
                            "⭐ Rating": breakdown.get("rating_score", 0),
                            "🔑 Keyword Match": breakdown.get("keyword_score", 0),
                        }
                        for label, val in scores_to_show.items():
                            st.markdown(f'<div class="score-bar-label">{label}: {val:.0f}%</div>', unsafe_allow_html=True)
                            st.progress(min(val / 100, 1.0))
                else:
                    st.info("Score breakdown not available")


def render_preferences_tags(prefs: Dict):
    """Render extracted preferences as colorful tags."""
    tags = []
    if prefs.get("category"):
        tags.append(f"📁 {prefs['category']}")
    if prefs.get("brand"):
        tags.append(f"🏷️ {prefs['brand']}")
    if prefs.get("max_price"):
        tags.append(f"💰 Under ₹{prefs['max_price']:,.0f}")
    if prefs.get("min_price"):
        tags.append(f"💰 Over ₹{prefs['min_price']:,.0f}")
    if prefs.get("min_rating"):
        tags.append(f"⭐ Min {prefs['min_rating']}★")
    if prefs.get("use_case"):
        tags.append(f"🎯 {prefs['use_case'].title()}")
    if prefs.get("budget_type"):
        tags.append(f"💸 {prefs['budget_type'].title()} budget")

    if tags:
        tag_html = "".join([f'<span class="pref-tag">{t}</span>' for t in tags])
        st.markdown(
            f'<div style="margin:0.5rem 0 1rem 0;"><strong style="font-size:0.85rem;color:#374151;">Detected preferences:</strong><br/>{tag_html}</div>',
            unsafe_allow_html=True,
        )


# ── Session state defaults ────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "last_prefs" not in st.session_state:
    st.session_state.last_prefs = None
if "searching" not in st.session_state:
    st.session_state.searching = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Filters")
    st.markdown('<div class="sidebar-header">Refine your search</div>', unsafe_allow_html=True)

    # Load categories/brands from API
    cats_data = api_get("/categories")
    categories = ["All"] + (cats_data.get("categories", []) if cats_data else [])
    brands = ["All"] + (cats_data.get("brands", []) if cats_data else [])

    selected_category = st.selectbox("📁 Category", categories, index=0)
    selected_brand = st.selectbox("🏷️ Brand", brands, index=0)
    max_price_filter = st.number_input(
        "💰 Max Price (₹)", min_value=0, max_value=500000, value=0, step=1000,
        help="Set to 0 for no limit"
    )
    min_rating_filter = st.slider("⭐ Minimum Rating", 0.0, 5.0, 0.0, 0.1)
    availability_filter = st.selectbox("📦 Availability", ["All", "In Stock", "Out of Stock"])

    st.markdown("---")

    # Health status
    health = api_get("/health")
    if health:
        st.markdown("**System Status**")
        st.markdown(f"🟢 **Products:** {health.get('products_loaded', 0)}")
        st.markdown(f"🧠 **Embeddings:** {health.get('embedding_model', 'unknown')}")
        llm_status = "🟢 Available" if health.get("llm_available") else "🟡 Rule-based fallback"
        st.markdown(f"🤖 **AI:** {llm_status}")

    st.markdown("---")

    # Recent searches
    st.markdown("**🕒 Recent Searches**")
    history = api_get("/history", params={"limit": 5})
    if history and history.get("history"):
        for item in history["history"]:
            ts = item.get("timestamp", "")[:16].replace("T", " ")
            q = item.get("query", "")[:45]
            st.markdown(
                f'<div class="history-item">🔍 {q}<br/><small style="color:#94a3b8;">{ts}</small></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"↩ Re-search", key=f"redo_{item.get('id', ts)}"):
                st.session_state.query = item.get("query", "")
                st.session_state.results = None
                st.rerun()
    else:
        st.info("No recent searches yet.")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="hero-title">🛍️ AI-Powered E-Commerce<br/>Recommendation System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-subtitle">Find products that match what you\'re actually looking for — powered by AI.</p>',
    unsafe_allow_html=True,
)

# Search box
col_search, col_btn = st.columns([5, 1])
with col_search:
    query_input = st.text_input(
        "search_query",
        value=st.session_state.query,
        placeholder='Try: "affordable Sony headphones for music under ₹5000"',
        label_visibility="collapsed",
        key="search_input",
    )
with col_btn:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# Example queries
st.markdown("**Quick examples:**")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for i, (col, ex) in enumerate(zip(example_cols, EXAMPLE_QUERIES)):
    with col:
        # Shorten for button display
        short = ex[:28] + "…" if len(ex) > 28 else ex
        if st.button(short, key=f"ex_{i}", help=ex, use_container_width=True):
            st.session_state.query = ex
            st.session_state.results = None
            st.rerun()

st.markdown("---")

# Trigger search
effective_query = query_input.strip()
if search_clicked and effective_query:
    payload = {
        "query": effective_query,
        "top_k": 5,
    }
    if selected_category and selected_category != "All":
        payload["category"] = selected_category
    if selected_brand and selected_brand != "All":
        payload["brand"] = selected_brand
    if max_price_filter and max_price_filter > 0:
        payload["max_price"] = float(max_price_filter)
    if min_rating_filter and min_rating_filter > 0:
        payload["min_rating"] = float(min_rating_filter)
    if availability_filter and availability_filter != "All":
        payload["availability"] = availability_filter

    with st.spinner("🔍 Searching and ranking products..."):
        result = api_post("/recommend", payload)
        st.session_state.results = result
        st.session_state.last_prefs = result.get("preferences") if result else None
        st.session_state.query = effective_query

# Session state search (from re-search buttons)
elif st.session_state.query and st.session_state.results is None and not search_clicked:
    # Auto-search when query is set via history re-search
    payload = {
        "query": st.session_state.query,
        "top_k": 5,
    }
    if selected_category and selected_category != "All":
        payload["category"] = selected_category
    if selected_brand and selected_brand != "All":
        payload["brand"] = selected_brand
    if max_price_filter and max_price_filter > 0:
        payload["max_price"] = float(max_price_filter)
    if min_rating_filter and min_rating_filter > 0:
        payload["min_rating"] = float(min_rating_filter)
    if availability_filter and availability_filter != "All":
        payload["availability"] = availability_filter

    with st.spinner("🔍 Searching and ranking products..."):
        result = api_post("/recommend", payload)
        st.session_state.results = result
        st.session_state.last_prefs = result.get("preferences") if result else None

# ── Results ───────────────────────────────────────────────────────────────────
results = st.session_state.results

if results is not None:
    recs = results.get("recommendations", [])
    prefs = results.get("preferences", {})
    total = results.get("total_products_considered", 0)
    overall_source = results.get("explanation_source", "rule-based")

    # Preferences
    render_preferences_tags(prefs)

    if recs:
        meta_col1, meta_col2 = st.columns([3, 1])
        with meta_col1:
            source_label = "✨ AI-powered" if overall_source == "ai" else "⚙️ Rule-based"
            st.markdown(
                f'<div class="section-heading">🎯 Recommended for you '
                f'<span style="font-size:0.75rem;font-weight:400;color:#64748b;">'
                f'— {len(recs)} results from {total} products · {source_label}</span></div>',
                unsafe_allow_html=True,
            )
        with meta_col2:
            if st.button("🔄 Clear results"):
                st.session_state.results = None
                st.session_state.query = ""
                st.rerun()

        for i, rec in enumerate(recs):
            render_product_card(rec, i)

    else:
        st.markdown(
            """
<div class="no-results">
    <h3 style="color:#64748b;">😕 No products matched your search</h3>
    <p style="color:#94a3b8;">Try increasing your budget, removing filters, or broadening your query.</p>
</div>
""",
            unsafe_allow_html=True,
        )

else:
    # Welcome state
    st.markdown(
        """
<div style="text-align:center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
     border-radius: 20px; border: 1px solid #e0e7ff; margin-top:1rem;">
    <div style="font-size:3.5rem; margin-bottom:0.75rem;">🔍</div>
    <h2 style="color:#1e293b; margin-bottom:0.5rem;">Start your AI-powered search</h2>
    <p style="color:#64748b; font-size:1rem; max-width:500px; margin:0 auto;">
        Type a natural language query like <strong>"affordable Sony headphones for music under ₹5000"</strong>
        and our AI will find the best matching products for you.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Feature highlights
    st.markdown("<br/>", unsafe_allow_html=True)
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    features = [
        ("🧠", "Natural Language", "Search in plain English — no forms to fill"),
        ("📊", "Hybrid Scoring", "Combines semantic similarity with smart rule-based scoring"),
        ("💡", "AI Explanations", "Understand exactly why each product is recommended"),
        ("🎯", "Smart Filters", "Combine natural language with sidebar filters"),
    ]
    for col, (icon, title, desc) in zip([feat_col1, feat_col2, feat_col3, feat_col4], features):
        with col:
            st.markdown(
                f"""
<div style="text-align:center; padding:1.25rem 0.75rem; background:#fff;
     border-radius:12px; border:1px solid #e2e8f0; height:100%;">
    <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
    <div style="font-weight:700; color:#1e293b; font-size:0.95rem;">{title}</div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )
