#!/bin/bash
# ─────────────────────────────────────────────────────────────
# AI E-Commerce Recommendation System — Run Script
# ─────────────────────────────────────────────────────────────
set -e
VENV_DIR=".venv"
REQUIREMENTS="backend/requirements.txt"

echo "🛍️  AI E-Commerce Recommendation System"
echo "========================================="

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/upgrade requirements
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$REQUIREMENTS"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application, run in TWO SEPARATE terminals:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    source $VENV_DIR/bin/activate"
echo "    uvicorn backend.main:app --reload --port 8000"
echo ""
echo "  Terminal 2 (Frontend):"
echo "    source $VENV_DIR/bin/activate"
echo "    streamlit run frontend/app.py"
echo ""
echo "Then open: http://localhost:8501"
echo ""

read -p "Start both now? (y/N): " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Starting backend..."
    uvicorn backend.main:app --reload --port 8000 &
    BACKEND_PID=$!
    sleep 3
    echo "Starting frontend..."
    streamlit run frontend/app.py &
    FRONTEND_PID=$!
    echo ""
    echo "🚀 Running!"
    echo "   Backend:  http://localhost:8000"
    echo "   Frontend: http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop both servers."
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
    wait
fi
