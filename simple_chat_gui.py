#!/usr/bin/env python3
"""
MiniLLM Simple Chat GUI - Clean and Minimal Interface

A simple, clean chat interface using pure tkinter with minimal styling.
Focus on functionality over fancy design.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import sys
import os

# Add the current directory to the path to import our chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_chatbot import UniversalChatbot

class SimpleChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.chatbot = UniversalChatbot()
        self.chat_mode = "general"  # "general" or "pdf"
        self.current_pdf = None
        
        self.setup_window()
        self.create_widgets()
        
        # Initialize chatbot in background
        threading.Thread(target=self.initialize_chatbot, daemon=True).start()
    
    def setup_window(self):
        """Configure the main window."""
        self.root.title("MiniLLM AI Assistant")
        self.root.geometry("900x700")
        self.root.minsize(600, 500)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"900x700+{x}+{y}")
        
        # Simple color scheme
        self.root.configure(bg='#f0f0f0')
    
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🤖 MiniLLM AI Assistant",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(0, 10))
        
        # Mode selection frame
        mode_frame = tk.Frame(main_frame, bg='#f0f0f0')
        mode_frame.pack(fill='x', pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="general")
        
        general_radio = tk.Radiobutton(
            mode_frame,
            text="General Chat",
            variable=self.mode_var,
            value="general",
            command=self.on_mode_change,
            bg='#f0f0f0',
            font=('Arial', 10)
        )
        general_radio.pack(side='left', padx=(0, 20))
        
        pdf_radio = tk.Radiobutton(
            mode_frame,
            text="PDF Chat",
            variable=self.mode_var,
            value="pdf",
            command=self.on_mode_change,
            bg='#f0f0f0',
            font=('Arial', 10)
        )
        pdf_radio.pack(side='left')
        
        # PDF upload frame
        pdf_frame = tk.Frame(main_frame, bg='#f0f0f0')
        pdf_frame.pack(fill='x', pady=(0, 10))
        
        self.upload_btn = tk.Button(
            pdf_frame,
            text="Upload PDF",
            command=self.upload_pdf,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10),
            relief='flat',
            padx=20,
            pady=5
        )
        self.upload_btn.pack(side='left')
        
        self.pdf_status_label = tk.Label(
            pdf_frame,
            text="No PDF loaded",
            bg='#f0f0f0',
            fg='#666666',
            font=('Arial', 9)
        )
        self.pdf_status_label.pack(side='left', padx=(10, 0))
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg='white', relief='sunken', bd=1)
        chat_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap='word',
            bg='white',
            fg='#333333',
            font=('Arial', 10),
            relief='flat',
            padx=10,
            pady=10,
            state='disabled'
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_configure('user', foreground='#1976D2', font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='#4CAF50', font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure('timestamp', foreground='#999999', font=('Arial', 8))
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg='#f0f0f0')
        input_frame.pack(fill='x')
        
        # Message input
        self.message_entry = tk.Text(
            input_frame,
            height=3,
            wrap='word',
            bg='white',
            fg='#333333',
            font=('Arial', 10),
            relief='sunken',
            bd=1,
            padx=5,
            pady=5
        )
        self.message_entry.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Send button
        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        )
        self.send_btn.pack(side='right')
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg='#f0f0f0')
        status_frame.pack(fill='x', pady=(5, 0))
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Initializing...",
            bg='#f0f0f0',
            fg='#666666',
            font=('Arial', 8),
            anchor='w'
        )
        self.status_label.pack(fill='x')
        
        # Bind Enter key to send message
        self.message_entry.bind('<Control-Return>', lambda e: self.send_message())
        
        # Add welcome message
        self.add_welcome_message()
    
    def add_welcome_message(self):
        """Add a welcome message to the chat."""
        welcome_text = """Welcome to MiniLLM AI Assistant!

I can help you with:
• General questions and conversations
• PDF document analysis (upload a PDF first)
• Information lookup and explanations

Choose your mode above and start chatting!"""
        
        self.add_message("assistant", welcome_text)
    
    def add_message(self, sender, message, timestamp=None):
        """Add a message to the chat display."""
        import datetime
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%H:%M")
        
        self.chat_display.config(state='normal')
        
        # Add sender and timestamp
        if sender == "user":
            self.chat_display.insert('end', f"You ({timestamp}):\n", 'user')
        else:
            self.chat_display.insert('end', f"Assistant ({timestamp}):\n", 'assistant')
        
        # Add message content
        self.chat_display.insert('end', f"{message}\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.see('end')
        self.chat_display.config(state='disabled')
    
    def initialize_chatbot(self):
        """Initialize the chatbot in a separate thread."""
        try:
            success = self.chatbot.setup_api_keys()
            if success:
                self.root.after(0, lambda: self.status_label.configure(
                    text="Status: Ready"
                ))
            else:
                self.root.after(0, lambda: self.status_label.configure(
                    text="Status: Error - Check API keys"
                ))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(
                text=f"Status: Error - {str(e)}"
            ))
    
    def on_mode_change(self):
        """Handle mode change between general and PDF chat."""
        new_mode = self.mode_var.get()
        
        if new_mode == "pdf" and not self.chatbot.pdf_loaded:
            messagebox.showinfo(
                "PDF Required",
                "Please upload a PDF document first to use PDF chat mode."
            )
            self.mode_var.set("general")
            return
        
        self.chat_mode = new_mode
        
        if new_mode == "general":
            self.status_label.configure(text="Status: General Chat Mode")
        else:
            pdf_name = Path(self.current_pdf).name if self.current_pdf else "Unknown"
            self.status_label.configure(text=f"Status: PDF Chat Mode - {pdf_name}")
    
    def upload_pdf(self):
        """Handle PDF upload."""
        file_path = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.status_label.configure(text="Status: Processing PDF...")
        self.upload_btn.configure(text="Processing...", state='disabled')
        
        def process_pdf():
            try:
                success = self.chatbot.process_pdf(file_path)
                
                if success:
                    self.current_pdf = file_path
                    pdf_name = Path(file_path).name
                    
                    self.root.after(0, lambda: self.pdf_status_label.configure(
                        text=f"Loaded: {pdf_name[:40]}{'...' if len(pdf_name) > 40 else ''}"
                    ))
                    self.root.after(0, lambda: self.status_label.configure(
                        text="Status: PDF loaded successfully"
                    ))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="Upload PDF", state='normal'
                    ))
                    
                    # Add success message
                    self.root.after(0, lambda: self.add_message(
                        "assistant",
                        f"Successfully loaded PDF: {pdf_name}\n\nYou can now switch to PDF Chat mode and ask questions about the document!"
                    ))
                    
                else:
                    self.root.after(0, lambda: self.status_label.configure(
                        text="Status: Error processing PDF"
                    ))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="Upload PDF", state='normal'
                    ))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Failed to process PDF. Please check the file and try again."
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(
                    text=f"Status: Error - {str(e)}"
                ))
                self.root.after(0, lambda: self.upload_btn.configure(
                    text="Upload PDF", state='normal'
                ))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"An error occurred: {str(e)}"
                ))
        
        threading.Thread(target=process_pdf, daemon=True).start()
    
    def send_message(self):
        """Send a message and get AI response."""
        message = self.message_entry.get('1.0', 'end-1c').strip()
        
        if not message:
            return
        
        # Clear input
        self.message_entry.delete('1.0', 'end')
        
        # Add user message
        self.add_message("user", message)
        
        # Update status
        self.status_label.configure(text="Status: Thinking...")
        
        def get_response():
            try:
                if self.chat_mode == "pdf" and self.chatbot.pdf_loaded:
                    response, _ = self.chatbot.chat_with_pdf(message)
                else:
                    response = self.chatbot.chat_general(message)
                
                self.root.after(0, lambda: self.add_message("assistant", response))
                self.root.after(0, lambda: self.status_label.configure(text="Status: Ready"))
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.root.after(0, lambda: self.add_message("assistant", error_msg))
                self.root.after(0, lambda: self.status_label.configure(text="Status: Error"))
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nGoodbye!")


def main():
    """Main function to run the Simple Chat GUI."""
    app = SimpleChatGUI()
    app.run()


if __name__ == "__main__":
    main()
