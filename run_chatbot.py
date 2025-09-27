#!/usr/bin/env python3
"""
Simple launcher script for the BNS Chatbot
This script provides an easy way to run the chatbot with minimal setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import gradio
        import cohere
        import pypdf
        import langchain
        import faiss
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements. Please run: pip install -r requirements.txt")
            return False

def main():
    """Main launcher function."""
    print("🏛️ BNS Chatbot Launcher")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not Path("bns_chatbot.py").exists():
        print("❌ bns_chatbot.py not found in current directory")
        print("Please run this script from the project directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check for PDF file
    pdf_files = list(Path(".").glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in current directory")
        print("Please ensure you have the BNS PDF file in this directory")
        
        pdf_path = input("Enter the full path to your BNS PDF file: ").strip()
        if not Path(pdf_path).exists():
            print("❌ PDF file not found. Exiting.")
            return
    
    print("🚀 Starting BNS Chatbot...")
    
    # Import and run the main application
    try:
        from bns_chatbot import main as run_chatbot
        run_chatbot()
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    main()
