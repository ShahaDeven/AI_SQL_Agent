#!/bin/bash
set -e

echo "============================================"
echo "  AI SQL Agent — Docker Startup"
echo "============================================"

# 1. Check if demo database exists, generate if missing
if [ ! -f "/app/data/sql_agent_demo.db" ] && [ ! -f "/app/data/supply_chain.db" ]; then
    echo ""
    echo "⚙️  No database found. Generating demo database (TPC-H SF=0.1)..."
    echo "   This only happens on first run."
    python /app/scripts/demo_db.py
    echo "✅ Demo database created."
else
    echo "✅ Database found."
fi

# 2. Check if ChromaDB vector store exists, generate if missing
if [ ! -d "/app/data/chroma_db" ]; then
    echo ""
    echo "⚙️  No vector store found. Building ChromaDB index..."
    echo "   This only happens on first run."
    python -c "from src.retriever import setup_vector_db; setup_vector_db()"
    echo "✅ Vector store created."
else
    echo "✅ Vector store found."
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "============================================"

# 3. Start the Streamlit app
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
