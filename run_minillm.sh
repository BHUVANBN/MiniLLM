#!/bin/bash

echo "🤖 MiniLLM Universal Chatbot Launcher (Linux/macOS)"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "launcher.py" ]; then
    echo "❌ launcher.py not found in current directory"
    echo "Please run this script from the project directory"
    exit 1
fi

# Check and install requirements if needed
echo "📦 Checking requirements..."
python3 -c "import cohere, pypdf, langchain, faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        echo "Please run: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "✅ Requirements satisfied"
echo "🚀 Starting MiniLLM Universal Chatbot..."
echo

# Run the launcher
python3 launcher.py
