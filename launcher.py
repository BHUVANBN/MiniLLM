#!/usr/bin/env python3
"""
Universal Chatbot Launcher

This script provides options to launch different interfaces:
1. Tkinter GUI for PDF chat with upload functionality
2. CLI for general AI chat without PDF
3. Original Gradio web interface (BNS-specific)
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("🤖 MiniLLM Universal AI Chatbot Launcher")
    print("=" * 60)
    print("Choose your preferred interface:")
    print()
    print("1. 🖥️  GUI App (Tkinter) - Upload any PDF and chat")
    print("2. 💻 CLI Chat - General AI conversation")
    print("3. 🌐 Web App (Gradio) - Web interface for any PDF")
    print("4. ❌ Exit")
    print("=" * 60)


def check_requirements():
    """Check if required packages are installed."""
    try:
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


def launch_gui():
    """Launch the Tkinter GUI application."""
    try:
        print("🚀 Launching GUI application...")
        from chatgpt_ui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Error importing GUI: {e}")
        print("Make sure chatgpt_ui.py is in the same directory")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")


def launch_cli():
    """Launch the CLI application."""
    try:
        print("🚀 Launching CLI application...")
        from chat_cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI: {e}")
        print("Make sure chat_cli.py is in the same directory")
    except Exception as e:
        print(f"❌ Error launching CLI: {e}")


def launch_web():
    """Launch the Gradio web application."""
    try:
        print("🚀 Launching web application...")
        from web_app import main as web_main
        web_main()
    except ImportError as e:
        print(f"❌ Error importing web app: {e}")
        print("Make sure web_app.py is in the same directory")
        print("📦 Install Gradio with: pip install gradio")
    except Exception as e:
        print(f"❌ Error launching web app: {e}")


def main():
    """Main launcher function."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Check if we're in the right directory
    required_files = ['universal_chatbot.py', 'chatgpt_ui.py', 'chat_cli.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run this script from the project directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    while True:
        print_banner()
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                launch_gui()
                break
            
            elif choice == '2':
                launch_cli()
                break
            
            elif choice == '3':
                launch_web()
                break
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                input("Press Enter to continue...")
                os.system('cls' if os.name == 'nt' else 'clear')
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
