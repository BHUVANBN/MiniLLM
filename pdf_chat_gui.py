#!/usr/bin/env python3
"""
PDF Chat GUI Application using Tkinter

A user-friendly GUI application that allows users to upload any PDF
and chat with it using AI, or use it as a general AI assistant.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import os

from universal_chatbot import UniversalChatbot


class PDFChatGUI:
    """GUI application for PDF chatbot using Tkinter."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("🤖 MiniLLM Universal PDF Chatbot")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize chatbot
        self.chatbot = UniversalChatbot()
        self.api_initialized = False
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize APIs in background
        self.initialize_apis()
    
    def setup_gui(self):
        """Setup the GUI components."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 MiniLLM Universal PDF Chatbot", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # PDF Section
        pdf_frame = ttk.LabelFrame(main_frame, text="📄 PDF Management", padding="10")
        pdf_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        pdf_frame.columnconfigure(1, weight=1)
        
        # PDF upload button
        self.upload_btn = ttk.Button(pdf_frame, text="📁 Upload PDF", 
                                    command=self.upload_pdf)
        self.upload_btn.grid(row=0, column=0, padx=(0, 10))
        
        # PDF info label
        self.pdf_info_var = tk.StringVar(value="No PDF loaded")
        self.pdf_info_label = ttk.Label(pdf_frame, textvariable=self.pdf_info_var)
        self.pdf_info_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Clear PDF button
        self.clear_btn = ttk.Button(pdf_frame, text="🗑️ Clear PDF", 
                                   command=self.clear_pdf, state='disabled')
        self.clear_btn.grid(row=0, column=2, padx=(10, 0))
        
        # Chat Section
        chat_frame = ttk.LabelFrame(main_frame, text="💬 Chat", padding="10")
        chat_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                                     height=20, font=('Arial', 10))
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                              pady=(0, 10))
        
        # Input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        # Message input
        self.message_var = tk.StringVar()
        self.message_entry = ttk.Entry(input_frame, textvariable=self.message_var, 
                                      font=('Arial', 10))
        self.message_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.message_entry.bind('<Return>', self.send_message)
        
        # Send button
        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_message,
                                  state='disabled')
        self.send_btn.grid(row=0, column=1)
        
        # Mode selection
        mode_frame = ttk.Frame(chat_frame)
        mode_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Label(mode_frame, text="Chat Mode:").grid(row=0, column=0, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="general")
        self.general_radio = ttk.Radiobutton(mode_frame, text="General Chat", 
                                           variable=self.mode_var, value="general")
        self.general_radio.grid(row=0, column=1, padx=(0, 10))
        
        self.pdf_radio = ttk.Radiobutton(mode_frame, text="PDF Chat", 
                                        variable=self.mode_var, value="pdf", state='disabled')
        self.pdf_radio.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing APIs...")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Add welcome message
        self.add_message("🤖 Assistant", 
                        "Welcome to MiniLLM Universal PDF Chatbot!\n\n"
                        "• Upload a PDF to chat about its contents\n"
                        "• Or use General Chat mode for any questions\n"
                        "• Initializing APIs, please wait...")
    
    def initialize_apis(self):
        """Initialize APIs in a separate thread."""
        def init_thread():
            try:
                success = self.chatbot.initialize_apis()
                self.root.after(0, self.on_api_init_complete, success)
            except Exception as e:
                self.root.after(0, self.on_api_init_complete, False, str(e))
        
        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
    
    def on_api_init_complete(self, success, error=None):
        """Handle API initialization completion."""
        if success:
            self.api_initialized = True
            self.send_btn.config(state='normal')
            self.status_var.set("Ready! You can start chatting.")
            self.add_message("🤖 Assistant", "APIs initialized successfully! You can now start chatting.")
        else:
            self.status_var.set(f"API initialization failed: {error or 'Unknown error'}")
            self.add_message("❌ Error", f"Failed to initialize APIs: {error or 'Unknown error'}")
    
    def upload_pdf(self):
        """Handle PDF upload."""
        if not self.api_initialized:
            messagebox.showwarning("Warning", "Please wait for API initialization to complete.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            self.process_pdf(file_path)
    
    def process_pdf(self, file_path):
        """Process PDF in a separate thread."""
        self.upload_btn.config(state='disabled')
        self.status_var.set("Processing PDF...")
        
        def process_thread():
            try:
                success = self.chatbot.process_pdf(file_path)
                self.root.after(0, self.on_pdf_processed, success, file_path)
            except Exception as e:
                self.root.after(0, self.on_pdf_processed, False, file_path, str(e))
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def on_pdf_processed(self, success, file_path, error=None):
        """Handle PDF processing completion."""
        self.upload_btn.config(state='normal')
        
        if success:
            filename = Path(file_path).name
            self.pdf_info_var.set(f"📄 {filename} (Ready)")
            self.clear_btn.config(state='normal')
            self.pdf_radio.config(state='normal')
            self.mode_var.set("pdf")
            self.status_var.set("PDF processed successfully!")
            self.add_message("🤖 Assistant", 
                           f"PDF '{filename}' processed successfully! "
                           f"You can now ask questions about its content.")
        else:
            self.status_var.set(f"PDF processing failed: {error or 'Unknown error'}")
            self.add_message("❌ Error", f"Failed to process PDF: {error or 'Unknown error'}")
    
    def clear_pdf(self):
        """Clear the current PDF."""
        self.chatbot.clear_pdf_context()
        self.pdf_info_var.set("No PDF loaded")
        self.clear_btn.config(state='disabled')
        self.pdf_radio.config(state='disabled')
        self.mode_var.set("general")
        self.status_var.set("PDF cleared. Ready for general chat.")
        self.add_message("🤖 Assistant", "PDF context cleared. Switched to general chat mode.")
    
    def send_message(self, event=None):
        """Send a message."""
        if not self.api_initialized:
            messagebox.showwarning("Warning", "Please wait for API initialization to complete.")
            return
        
        message = self.message_var.get().strip()
        if not message:
            return
        
        # Add user message to chat
        self.add_message("👤 You", message)
        self.message_var.set("")
        
        # Disable send button temporarily
        self.send_btn.config(state='disabled')
        self.status_var.set("Generating response...")
        
        # Generate response in separate thread
        def response_thread():
            try:
                if self.mode_var.get() == "pdf" and self.chatbot.pdf_loaded:
                    response, source = self.chatbot.chat_with_pdf(message)
                    self.root.after(0, self.on_response_received, response, source)
                else:
                    response = self.chatbot.chat_general(message)
                    self.root.after(0, self.on_response_received, response)
            except Exception as e:
                self.root.after(0, self.on_response_received, f"Error: {str(e)}")
        
        thread = threading.Thread(target=response_thread, daemon=True)
        thread.start()
    
    def on_response_received(self, response, source=None):
        """Handle response received."""
        self.send_btn.config(state='normal')
        self.status_var.set("Ready")
        
        # Add response to chat
        self.add_message("🤖 Assistant", response)
        
        # Add source if available
        if source and source.strip():
            self.add_message("📚 Source", source, color='gray')
    
    def add_message(self, sender, message, color='black'):
        """Add a message to the chat display."""
        self.chat_display.config(state='normal')
        
        # Add sender with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M")
        
        self.chat_display.insert(tk.END, f"\n[{timestamp}] {sender}:\n", 'sender')
        self.chat_display.insert(tk.END, f"{message}\n", 'message')
        
        # Configure tags for styling
        self.chat_display.tag_config('sender', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('message', font=('Arial', 10), foreground=color)
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = PDFChatGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")


if __name__ == "__main__":
    main()
