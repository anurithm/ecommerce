"""
Embeddings module with sentence-transformers and TF-IDF fallback.
Caches product embeddings for performance.
"""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR.mkdir(exist_ok=True)
EMBEDDING_CACHE_FILE = CACHE_DIR / "product_embeddings.pkl"

# Global state
_embedding_model = None
_use_sentence_transformers = False
_tfidf_vectorizer = None


def _get_embedding_model():
    """Load and cache the sentence-transformers model."""
    global _embedding_model, _use_sentence_transformers

    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _use_sentence_transformers = True
        logger.info("sentence-transformers model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load sentence-transformers: {e}. Using TF-IDF fallback.")
        _embedding_model = None
        _use_sentence_transformers = False

    return _embedding_model


def get_model_type() -> str:
    """Return the embedding model type being used."""
    _get_embedding_model()
    return "sentence-transformers" if _use_sentence_transformers else "tf-idf"


def create_product_text(product: pd.Series) -> str:
    """Create a rich text representation of a product for embedding."""
    parts = []

    if pd.notna(product.get("product_name")):
        parts.append(str(product["product_name"]))
    if pd.notna(product.get("brand")):
        parts.append(f"Brand: {product['brand']}")
    if pd.notna(product.get("category")):
        parts.append(f"Category: {product['category']}")
    if pd.notna(product.get("subcategory")):
        parts.append(f"Type: {product['subcategory']}")
    if pd.notna(product.get("description")):
        parts.append(str(product["description"]))
    if pd.notna(product.get("features")):
        parts.append(f"Features: {product['features']}")
    if pd.notna(product.get("color")):
        parts.append(f"Color: {product['color']}")

    return " ".join(parts)


def _compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the DataFrame to detect changes."""
    content = df.to_json()
    return hashlib.md5(content.encode()).hexdigest()


def _load_embedding_cache(data_hash: str) -> Optional[Tuple[np.ndarray, str]]:
    """Load cached embeddings if they exist and match the data hash."""
    if EMBEDDING_CACHE_FILE.exists():
        try:
            with open(EMBEDDING_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("data_hash") == data_hash:
                logger.info("Loaded product embeddings from cache")
                return cache["embeddings"], cache["model_type"]
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
    return None


def _save_embedding_cache(embeddings: np.ndarray, data_hash: str, model_type: str) -> None:
    """Save embeddings to cache."""
    try:
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump(
                {"embeddings": embeddings, "data_hash": data_hash, "model_type": model_type},
                f,
            )
        logger.info("Saved product embeddings to cache")
    except Exception as e:
        logger.warning(f"Could not save embedding cache: {e}")


def get_product_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    Get product embeddings, using cache if available.
    Returns (embeddings, model_type)
    """
    global _tfidf_vectorizer

    product_texts = df.apply(create_product_text, axis=1).tolist()
    data_hash = _compute_data_hash(df)

    # Check cache
    cached = _load_embedding_cache(data_hash)
    if cached is not None:
        embeddings, model_type = cached
        if model_type == "tf-idf" and _tfidf_vectorizer is None:
            # Need to rebuild TF-IDF vectorizer for query embedding
            _rebuild_tfidf(product_texts)
        return embeddings, model_type

    # Generate new embeddings
    model = _get_embedding_model()

    if _use_sentence_transformers and model is not None:
        logger.info(f"Generating embeddings for {len(product_texts)} products...")
        embeddings = model.encode(product_texts, show_progress_bar=False, batch_size=32)
        embeddings = embeddings.astype(np.float32)
        model_type = "sentence-transformers"
    else:
        logger.info("Generating TF-IDF embeddings...")
        embeddings, model_type = _tfidf_embed(product_texts)

    _save_embedding_cache(embeddings, data_hash, model_type)
    return embeddings, model_type


def _rebuild_tfidf(texts: List[str]) -> None:
    """Rebuild TF-IDF vectorizer from texts."""
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    _tfidf_vectorizer.fit(texts)


def _tfidf_embed(texts: List[str]) -> Tuple[np.ndarray, str]:
    """Generate TF-IDF embeddings."""
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    embeddings = _tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return embeddings, "tf-idf"


def embed_query(query: str) -> np.ndarray:
    """Embed a user query using the same model as products."""
    global _tfidf_vectorizer

    model = _get_embedding_model()

    if _use_sentence_transformers and model is not None:
        vec = model.encode([query], show_progress_bar=False)
        return vec[0].astype(np.float32)
    else:
        # Use TF-IDF
        if _tfidf_vectorizer is None:
            logger.warning("TF-IDF vectorizer not initialized. Using zero vector.")
            return np.zeros(5000, dtype=np.float32)
        vec = _tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
        return vec[0]


def cosine_similarity_single(vec_a: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single vector and a matrix of vectors."""
    norm_a = np.linalg.norm(vec_a)
    if norm_a == 0:
        return np.zeros(len(matrix))

    norms_b = np.linalg.norm(matrix, axis=1)
    # Avoid division by zero
    norms_b = np.where(norms_b == 0, 1e-10, norms_b)

    similarities = np.dot(matrix, vec_a) / (norms_b * norm_a)
    return np.clip(similarities, 0, 1)
