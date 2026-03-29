#!/bin/bash
echo "🚀 Starting MiniLLM Offline AI Assistant..."

# Check if requirements are installed
if ! python3 -c "import langchain, chromadb, langchain_ollama" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Launch GUI
python3 chatgpt_ui.py
