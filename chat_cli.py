#!/usr/bin/env python3
"""
CLI Chat Interface

A command-line interface for chatting with the AI assistant
without any PDF context - just general conversation.
"""

import os
import sys
from pathlib import Path
import argparse

from universal_chatbot import UniversalChatbot


class ChatCLI:
    """Command-line interface for general AI chat."""
    
    def __init__(self):
        """Initialize the CLI chat interface."""
        self.chatbot = UniversalChatbot()
        self.running = False
    
    def print_banner(self):
        """Print the application banner."""
        print("=" * 60)
        print("🤖 MiniLLM Universal AI Chat - Command Line Interface")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' or 'exit' to leave")
        print("=" * 60)
    
    def print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
  help          - Show this help message
  clear         - Clear the screen
  quit/exit     - Exit the application
  
Just type your message and press Enter to chat with the AI!

Examples:
  > What is the capital of France?
  > Explain quantum computing in simple terms
  > Write a short poem about nature
  > Help me plan a weekend trip
"""
        print(help_text)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def initialize(self):
        """Initialize the chatbot APIs."""
        print("🔧 Initializing AI APIs...")
        
        if not self.chatbot.initialize_apis():
            print("❌ Failed to initialize APIs. Please check your API keys.")
            return False
        
        print("✅ APIs initialized successfully!")
        return True
    
    def chat_loop(self):
        """Main chat loop."""
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.running = False
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_screen()
                    self.print_banner()
                    continue
                
                # Generate AI response
                print("🤖 Assistant: ", end="", flush=True)
                
                try:
                    response = self.chatbot.chat_general(user_input)
                    print(response)
                
                except Exception as e:
                    print(f"❌ Error generating response: {e}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                self.running = False
                break
            
            except EOFError:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                self.running = False
                break
    
    def run(self):
        """Run the CLI application."""
        self.clear_screen()
        self.print_banner()
        
        if not self.initialize():
            return
        
        print("\n🚀 Ready to chat! Ask me anything...")
        self.chat_loop()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MiniLLM Universal AI Chat - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat_cli.py                    # Start interactive chat
  python chat_cli.py --help            # Show this help
  
Interactive Commands:
  help          - Show available commands
  clear         - Clear the screen
  quit/exit     - Exit the application
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='MiniLLM Universal AI Chat CLI v1.0'
    )
    
    args = parser.parse_args()
    
    # Run the CLI
    cli = ChatCLI()
    cli.run()


if __name__ == "__main__":
    main()
