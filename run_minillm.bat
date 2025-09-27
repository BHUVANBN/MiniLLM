@echo off
echo 🤖 MiniLLM Universal Chatbot Launcher (Windows)
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "bns_chatbot.py" (
    echo ❌ bns_chatbot.py not found in current directory
    echo Please run this script from the project directory
    pause
    exit /b 1
)

REM Install requirements if needed
echo 📦 Checking requirements...
python -c "import gradio, cohere, pypdf, langchain, faiss" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

echo ✅ Requirements satisfied
echo 🚀 Starting MiniLLM Universal Chatbot...
echo.

REM Run the launcher
python launcher.py

pause
