# AI-Powered E-Commerce Recommendation System 🛍️

A complete, production-ready full-stack e-commerce recommendation system powered by a hybrid rule-based and semantic search engine, and an LLM for natural language explanations.

## 📖 About the Project
This project is an advanced e-commerce product recommendation platform that goes beyond standard faceted search. By leveraging both semantic similarity and rule-based preference matching, it provides highly relevant product recommendations to users. Furthermore, it integrates a Large Language Model (LLM) to offer natural, personalized conversational explanations as to why specific products are matched with the user's query.

## 🛠 Tech Stacks Used
- **Frontend**: Streamlit (for building a highly interactive and responsive web UI)
- **Backend**: FastAPI (asynchronous, robust REST API)
- **Database**: SQLite (for persistent user history and preference tracking)
- **Machine Learning & NLP**: `sentence-transformers` for embedding generation, Scikit-learn for TF-IDF fallback 
- **LLM Integration**: OpenRouter API for generating conversational product explanations

## 🚀 Live Demo
You can view the live application here: *[Insert Streamlit Live Demo URL Here]*


## 🌟 Features

- **Hybrid Recommendation Engine:** Combines semantic search (`sentence-transformers`), rule-based preference extraction (category, brand, budget, ratings), and keyword scoring.
- **Natural Language Search:** Users can search like a human (e.g., *"affordable noise-cancelling headphones from Sony under ₹20,000"*).
- **AI Explanations:** Uses OpenRouter API to generate personalized, conversational explanations of *why* a product was recommended. Gracefully falls back to rule-based explanations if offline or API is unavailable.
- **FastAPI Backend:** Fully asynchronous, with SQLite for user history tracking and preference persistence.
- **Streamlit Frontend:** A modern, highly responsive frontend with custom CSS for a premium UI experience.

---

## 🚀 Quick Setup

### 1. Prerequisites
- **Python 3.10+**
- (Optional but required for AI explanations) get an **OpenRouter API Key** from [openrouter.ai](https://openrouter.ai).

### 2. Environment Setup

Copy `.env.example` to `.env` and insert your API key:
```bash
cp .env.example .env
# Edit .env with your favorite editor
```

### 3. Run the complete stack

Run the provided single startup script:
```bash
chmod +x run.sh
./run.sh
```
This script will automatically:
1. Create a Python virtual environment (`.venv`)
2. Install all dependencies from `backend/requirements.txt`
3. Optionally launch both the FastAPI backend and Streamlit frontend.

---

## 📚 Project Structure

```bash
ecommerce/
├── backend/
│   ├── models.py        # Pydantic data models
│   ├── database.py      # SQLite db for tracking search history
│   ├── embeddings.py    # sentence-transformers & TF-IDF caching layer
│   ├── recommender.py   # Hybrid rule-based & semantic engine
│   ├── llm.py           # OpenRouter async client
│   └── main.py          # FastAPI application & endpoints
├── frontend/
│   └── app.py           # Streamlit UI & dashboard
├── data/
│   └── products.csv     # 100+ realistic e-commerce products
├── models/              # Local cache directory for HF embeddings
├── tests/               # Pytest suite
│   ├── test_recommender.py
│   └── test_api.py
├── run.sh               # Quickstart script
└── README.md
```

## Running commands
1) Create/activate the virtual environment
     python3 -m venv .venv
     source .venv/bin/activate
2) Install the requirements
      pip install -r requirements.txt
3) Check the Streamlit app
      streamlit run frontend/app.py

## 🧪 Testing

The codebase includes comprehensive unit and integration tests covering natural language extraction, hybrid scoring logic, product parsing, and API endpoints.

To run the test suite:
```bash
source .venv/bin/activate
pytest tests/ -v
```

## 🛠 Features & Fallback Mechanisms

Built with robustness in mind:
- **No API Key?** The engine falls back to a deterministic, human-readable rule-based explanation.
- **No Internet/HF Down?** The embedding module falls back to TF-IDF `scikit-learn` representations.
- **Caching:** Expensive embeddings are hashed and saved to `models/cache.npz` to speed up subsequent dev startups.

Enjoy!
