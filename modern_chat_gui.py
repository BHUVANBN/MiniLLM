#!/usr/bin/env python3
"""
MiniLLM Modern Chat GUI - ChatGPT-inspired Material Design Interface

A beautiful, modern chat interface inspired by ChatGPT and Google Material Design.
Features a clean, professional UI with smooth animations and intuitive controls.
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

class ModernChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.chatbot = UniversalChatbot()
        self.chat_mode = "general"  # "general" or "pdf"
        self.current_pdf = None
        
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.setup_bindings()
        
        # Initialize chatbot in background
        threading.Thread(target=self.initialize_chatbot, daemon=True).start()
    
    def setup_window(self):
        """Configure the main window with modern styling."""
        self.root.title("🤖 MiniLLM - AI Assistant")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
        # Configure colors - Dark Material Design Theme
        self.colors = {
            'primary': '#BB86FC',       # Material Purple (Dark theme primary)
            'primary_dark': '#9C27B0',  # Darker purple
            'primary_light': '#E1BEE7', # Lighter purple
            'secondary': '#03DAC6',     # Material Teal (Dark theme secondary)
            'background': '#121212',    # Dark background
            'surface': '#1E1E1E',       # Dark surface
            'surface_variant': '#2D2D2D', # Darker surface variant
            'surface_elevated': '#252525', # Elevated surface
            'on_surface': '#E0E0E0',    # Light text on dark surface
            'on_surface_variant': '#A0A0A0', # Muted light text
            'on_primary': '#000000',    # Dark text on primary
            'outline': '#404040',       # Dark outline
            'success': '#4CAF50',       # Green (works in dark)
            'error': '#CF6679',         # Material error (dark theme)
            'warning': '#FF9800',       # Orange warning
            'user_bubble': '#BB86FC',   # Purple for user messages
            'user_text': '#000000',     # Dark text on purple
            'ai_bubble': '#2D2D2D',     # Dark grey for AI messages
            'ai_text': '#E0E0E0',       # Light text on dark
            'input_bg': '#252525',      # Input background
            'hover': '#333333',         # Hover state
            'shadow': '#000000',        # Shadow color
        }
        
        self.root.configure(bg=self.colors['background'])
    
    def setup_styles(self):
        """Configure ttk styles for modern appearance."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure button styles
        self.style.configure(
            'Primary.TButton',
            background=self.colors['primary'],
            foreground=self.colors['on_primary'],
            borderwidth=0,
            focuscolor='none',
            padding=(20, 12),
            font=('Segoe UI', 10, 'bold'),
            relief='flat'
        )
        
        self.style.map(
            'Primary.TButton',
            background=[('active', self.colors['primary_dark']),
                       ('pressed', self.colors['primary_dark'])]
        )
        
        # Configure secondary button style
        self.style.configure(
            'Secondary.TButton',
            background=self.colors['surface_variant'],
            foreground=self.colors['on_surface'],
            borderwidth=1,
            relief='flat',
            focuscolor='none',
            padding=(15, 10),
            font=('Segoe UI', 9, 'bold')
        )
        
        self.style.map(
            'Secondary.TButton',
            background=[('active', self.colors['hover']),
                       ('pressed', self.colors['hover'])]
        )
        
        # Configure frame styles
        self.style.configure(
            'Card.TFrame',
            background=self.colors['surface'],
            relief='flat',
            borderwidth=0
        )
        
        self.style.configure(
            'Sidebar.TFrame',
            background=self.colors['surface_elevated'],
            relief='flat',
            borderwidth=0
        )
        
        # Configure label styles
        self.style.configure(
            'Title.TLabel',
            background=self.colors['surface'],
            foreground=self.colors['on_surface'],
            font=('Segoe UI', 18, 'bold')
        )
        
        self.style.configure(
            'Subtitle.TLabel',
            background=self.colors['surface'],
            foreground=self.colors['on_surface_variant'],
            font=('Segoe UI', 11)
        )
        
        # Configure separator
        self.style.configure(
            'Dark.TSeparator',
            background=self.colors['outline']
        )
    
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_frame)
        
        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Left sidebar for controls
        self.create_sidebar(content_frame)
        
        # Main chat area
        self.create_chat_area(content_frame)
    
    def create_header(self, parent):
        """Create the header section with title and mode selector."""
        header_frame = ttk.Frame(parent, style='Card.TFrame')
        header_frame.pack(fill='x', pady=(0, 10))
        
        # Add subtle shadow effect with padding
        header_content = ttk.Frame(header_frame)
        header_content.pack(fill='x', padx=20, pady=15)
        
        # Title and subtitle
        title_frame = ttk.Frame(header_content)
        title_frame.pack(side='left', fill='x', expand=True)
        
        title_label = ttk.Label(
            title_frame,
            text="🤖 MiniLLM AI Assistant",
            style='Title.TLabel'
        )
        title_label.pack(anchor='w')
        
        self.subtitle_label = ttk.Label(
            title_frame,
            text="General AI Assistant - Ready to help!",
            style='Subtitle.TLabel'
        )
        self.subtitle_label.pack(anchor='w', pady=(2, 0))
        
        # Mode selector
        mode_frame = ttk.Frame(header_content)
        mode_frame.pack(side='right')
        
        self.mode_var = tk.StringVar(value="general")
        
        general_btn = ttk.Radiobutton(
            mode_frame,
            text="💬 General Chat",
            variable=self.mode_var,
            value="general",
            command=self.on_mode_change,
            style='Secondary.TButton'
        )
        general_btn.pack(side='left', padx=(0, 10))
        
        pdf_btn = ttk.Radiobutton(
            mode_frame,
            text="📄 PDF Chat",
            variable=self.mode_var,
            value="pdf",
            command=self.on_mode_change,
            style='Secondary.TButton'
        )
        pdf_btn.pack(side='left')
    
    def create_sidebar(self, parent):
        """Create the left sidebar with PDF controls."""
        sidebar_frame = ttk.Frame(parent, style='Sidebar.TFrame')
        sidebar_frame.pack(side='left', fill='y', padx=(0, 20))
        
        # Add subtle border effect
        border_frame = tk.Frame(
            sidebar_frame,
            bg=self.colors['outline'],
            width=1
        )
        border_frame.pack(side='right', fill='y')
        
        sidebar_content = ttk.Frame(sidebar_frame, style='Sidebar.TFrame')
        sidebar_content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # PDF Section
        pdf_label = tk.Label(
            sidebar_content,
            text="📄 PDF Documents",
            font=('Segoe UI', 13, 'bold'),
            fg=self.colors['on_surface'],
            bg=self.colors['surface_elevated']
        )
        pdf_label.pack(anchor='w', pady=(0, 15))
        
        # Upload button
        self.upload_btn = ttk.Button(
            sidebar_content,
            text="📁 Upload PDF",
            command=self.upload_pdf,
            style='Primary.TButton'
        )
        self.upload_btn.pack(fill='x', pady=(0, 15))
        
        # Current PDF info with card styling
        self.pdf_info_frame = tk.Frame(
            sidebar_content,
            bg=self.colors['surface_variant'],
            relief='flat',
            bd=0
        )
        self.pdf_info_frame.pack(fill='x', pady=(0, 20))
        
        self.pdf_status_label = tk.Label(
            self.pdf_info_frame,
            text="No PDF loaded",
            font=('Segoe UI', 10),
            fg=self.colors['on_surface_variant'],
            bg=self.colors['surface_variant'],
            padx=12,
            pady=8
        )
        self.pdf_status_label.pack(anchor='w', fill='x')
        
        # Separator
        separator = ttk.Separator(sidebar_content, orient='horizontal', style='Dark.TSeparator')
        separator.pack(fill='x', pady=20)
        
        # Quick actions
        actions_label = tk.Label(
            sidebar_content,
            text="⚡ Quick Actions",
            font=('Segoe UI', 13, 'bold'),
            fg=self.colors['on_surface'],
            bg=self.colors['surface_elevated']
        )
        actions_label.pack(anchor='w', pady=(0, 15))
        
        clear_btn = ttk.Button(
            sidebar_content,
            text="🗑️ Clear Chat",
            command=self.clear_chat,
            style='Secondary.TButton'
        )
        clear_btn.pack(fill='x', pady=(0, 10))
        
        # Status indicator
        self.status_frame = ttk.Frame(sidebar_content)
        self.status_frame.pack(side='bottom', fill='x', pady=(20, 0))
        
        self.status_indicator = tk.Canvas(
            self.status_frame,
            width=12,
            height=12,
            highlightthickness=0,
            bg=self.colors['surface_elevated']
        )
        self.status_indicator.pack(side='left', padx=(0, 10))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Initializing...",
            font=('Segoe UI', 9),
            fg=self.colors['on_surface_variant'],
            bg=self.colors['surface_elevated']
        )
        self.status_label.pack(side='left')
        
        # Set initial status
        self.update_status("initializing")
    
    def create_chat_area(self, parent):
        """Create the main chat area with modern message bubbles."""
        chat_frame = ttk.Frame(parent, style='Card.TFrame')
        chat_frame.pack(side='right', fill='both', expand=True)
        
        # Chat messages area
        self.create_messages_area(chat_frame)
        
        # Input area
        self.create_input_area(chat_frame)
    
    def create_messages_area(self, parent):
        """Create the scrollable messages area."""
        messages_frame = ttk.Frame(parent)
        messages_frame.pack(fill='both', expand=True, padx=20, pady=(20, 10))
        
        # Create canvas for custom scrolling
        self.canvas = tk.Canvas(
            messages_frame,
            bg=self.colors['background'],
            highlightthickness=0
        )
        
        self.scrollbar = ttk.Scrollbar(
            messages_frame,
            orient="vertical",
            command=self.canvas.yview
        )
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Welcome message
        self.add_welcome_message()
    
    def create_input_area(self, parent):
        """Create the message input area."""
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Input container with modern dark styling
        input_container = tk.Frame(
            input_frame,
            bg=self.colors['input_bg'],
            relief='flat',
            bd=0
        )
        input_container.pack(fill='x')
        
        # Add subtle border
        border_top = tk.Frame(input_container, bg=self.colors['outline'], height=1)
        border_top.pack(fill='x')
        
        # Input content frame
        input_content = tk.Frame(input_container, bg=self.colors['input_bg'])
        input_content.pack(fill='x', padx=20, pady=15)
        
        # Text input with dark theme
        self.message_entry = tk.Text(
            input_content,
            height=3,
            wrap='word',
            bg=self.colors['input_bg'],
            fg=self.colors['on_surface'],
            font=('Segoe UI', 12),
            relief='flat',
            bd=0,
            padx=15,
            pady=12,
            insertbackground=self.colors['primary'],  # Cursor color
            selectbackground=self.colors['primary'],  # Selection color
            selectforeground=self.colors['on_primary']
        )
        self.message_entry.pack(side='left', fill='both', expand=True)
        
        # Send button with enhanced styling
        send_btn_frame = tk.Frame(input_content, bg=self.colors['input_bg'])
        send_btn_frame.pack(side='right', padx=(15, 0), pady=5)
        
        self.send_btn = tk.Button(
            send_btn_frame,
            text="➤",
            command=self.send_message,
            bg=self.colors['primary'],
            fg=self.colors['on_primary'],
            font=('Segoe UI', 16, 'bold'),
            relief='flat',
            bd=0,
            width=3,
            height=2,
            cursor='hand2',
            activebackground=self.colors['primary_dark'],
            activeforeground=self.colors['on_primary']
        )
        self.send_btn.pack()
        
        # Placeholder text
        self.add_placeholder()
    
    def setup_bindings(self):
        """Setup keyboard bindings and events."""
        # Send message on Ctrl+Enter
        self.message_entry.bind('<Control-Return>', lambda e: self.send_message())
        
        # Focus events for placeholder
        self.message_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.message_entry.bind('<FocusOut>', self.on_entry_focus_out)
        
        # Enhanced button hover effects
        self.send_btn.bind('<Enter>', lambda e: self.send_btn.configure(
            bg=self.colors['primary_dark'],
            relief='raised'
        ))
        self.send_btn.bind('<Leave>', lambda e: self.send_btn.configure(
            bg=self.colors['primary'],
            relief='flat'
        ))
    
    def add_placeholder(self):
        """Add placeholder text to the input field."""
        self.message_entry.insert('1.0', "Type your message here... (Ctrl+Enter to send)")
        self.message_entry.configure(fg=self.colors['on_surface_variant'])
        self.placeholder_active = True
    
    def on_entry_focus_in(self, event):
        """Handle focus in event for placeholder."""
        if self.placeholder_active:
            self.message_entry.delete('1.0', 'end')
            self.message_entry.configure(fg=self.colors['on_surface'])
            self.placeholder_active = False
    
    def on_entry_focus_out(self, event):
        """Handle focus out event for placeholder."""
        if not self.message_entry.get('1.0', 'end-1c').strip():
            self.add_placeholder()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_welcome_message(self):
        """Add a welcome message to the chat."""
        welcome_text = """👋 Welcome to MiniLLM AI Assistant!

I'm your intelligent companion, ready to assist you with:

🧠 General questions and conversations
📄 PDF document analysis and Q&A  
🔍 Information lookup and explanations
💡 Creative writing and brainstorming

✨ Choose your mode above and let's start our conversation!"""
        
        self.add_message("assistant", welcome_text)
    
    def add_message(self, sender, message, timestamp=None):
        """Add a message bubble to the chat."""
        import datetime
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%H:%M")
        
        # Message container
        msg_container = ttk.Frame(self.scrollable_frame)
        msg_container.pack(fill='x', padx=10, pady=5)
        
        if sender == "user":
            # User message (right-aligned, purple)
            msg_frame = tk.Frame(
                msg_container,
                bg=self.colors['user_bubble'],
                relief='flat',
                bd=0
            )
            msg_frame.pack(side='right', anchor='e', padx=(80, 0), pady=5)
            
            # Add rounded corner effect with padding
            msg_content = tk.Frame(msg_frame, bg=self.colors['user_bubble'])
            msg_content.pack(padx=2, pady=2)
            
            # Message text
            msg_label = tk.Label(
                msg_content,
                text=message,
                bg=self.colors['user_bubble'],
                fg=self.colors['user_text'],
                font=('Segoe UI', 11),
                wraplength=450,
                justify='left',
                padx=18,
                pady=12
            )
            msg_label.pack()
            
            # Timestamp
            time_label = tk.Label(
                msg_container,
                text=f"You • {timestamp}",
                bg=self.colors['background'],
                fg=self.colors['on_surface_variant'],
                font=('Segoe UI', 9)
            )
            time_label.pack(side='right', anchor='e', padx=(0, 8), pady=(0, 5))
            
        else:
            # Assistant message (left-aligned, dark grey)
            msg_frame = tk.Frame(
                msg_container,
                bg=self.colors['ai_bubble'],
                relief='flat',
                bd=0
            )
            msg_frame.pack(side='left', anchor='w', padx=(0, 80), pady=5)
            
            # Add rounded corner effect with padding
            msg_content = tk.Frame(msg_frame, bg=self.colors['ai_bubble'])
            msg_content.pack(padx=2, pady=2)
            
            # AI icon and message container
            content_frame = tk.Frame(msg_content, bg=self.colors['ai_bubble'])
            content_frame.pack(fill='x', padx=18, pady=12)
            
            # AI icon with enhanced styling
            icon_label = tk.Label(
                content_frame,
                text="🤖",
                bg=self.colors['ai_bubble'],
                font=('Segoe UI', 14)
            )
            icon_label.pack(side='left', anchor='n', padx=(0, 12))
            
            # Message text with better styling
            msg_label = tk.Label(
                content_frame,
                text=message,
                bg=self.colors['ai_bubble'],
                fg=self.colors['ai_text'],
                font=('Segoe UI', 11),
                wraplength=480,
                justify='left'
            )
            msg_label.pack(side='left', fill='x', expand=True)
            
            # Timestamp
            time_label = tk.Label(
                msg_container,
                text=f"Assistant • {timestamp}",
                bg=self.colors['background'],
                fg=self.colors['on_surface_variant'],
                font=('Segoe UI', 9)
            )
            time_label.pack(side='left', anchor='w', padx=(8, 0), pady=(0, 5))
        
        # Auto-scroll to bottom
        self.root.after(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the chat to the bottom."""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def initialize_chatbot(self):
        """Initialize the chatbot in a separate thread."""
        try:
            success = self.chatbot.setup_api_keys()
            if success:
                self.root.after(0, lambda: self.update_status("ready"))
                self.root.after(0, lambda: self.subtitle_label.configure(
                    text="Ready to chat! Choose your mode above."
                ))
            else:
                self.root.after(0, lambda: self.update_status("error"))
                self.root.after(0, lambda: self.subtitle_label.configure(
                    text="Error: Failed to initialize. Check API keys."
                ))
        except Exception as e:
            self.root.after(0, lambda: self.update_status("error"))
            self.root.after(0, lambda: self.subtitle_label.configure(
                text=f"Error: {str(e)}"
            ))
    
    def update_status(self, status):
        """Update the status indicator."""
        self.status_indicator.delete("all")
        
        if status == "ready":
            color = self.colors['success']
            text = "Ready"
        elif status == "processing":
            color = self.colors['secondary']
            text = "Processing..."
        elif status == "error":
            color = self.colors['error']
            text = "Error"
        else:  # initializing
            color = self.colors['on_surface_variant']
            text = "Initializing..."
        
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline="")
        self.status_label.configure(text=text)
    
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
            self.subtitle_label.configure(text="General AI Assistant - Ask me anything!")
        else:
            pdf_name = Path(self.current_pdf).name if self.current_pdf else "Unknown"
            self.subtitle_label.configure(text=f"PDF Chat Mode - Discussing: {pdf_name}")
    
    def upload_pdf(self):
        """Handle PDF upload."""
        file_path = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.update_status("processing")
        self.upload_btn.configure(text="📄 Processing...", state='disabled')
        
        def process_pdf():
            try:
                success = self.chatbot.process_pdf(file_path)
                
                if success:
                    self.current_pdf = file_path
                    pdf_name = Path(file_path).name
                    
                    self.root.after(0, lambda: self.pdf_status_label.configure(
                        text=f"✅ {pdf_name[:30]}{'...' if len(pdf_name) > 30 else ''}"
                    ))
                    self.root.after(0, lambda: self.update_status("ready"))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="📁 Upload PDF", state='normal'
                    ))
                    
                    # Add success message
                    self.root.after(0, lambda: self.add_message(
                        "assistant",
                        f"✅ Successfully loaded PDF: {pdf_name}\n\nYou can now switch to PDF Chat mode and ask questions about the document!"
                    ))
                    
                else:
                    self.root.after(0, lambda: self.update_status("error"))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="📁 Upload PDF", state='normal'
                    ))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Failed to process PDF. Please check the file and try again."
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: self.update_status("error"))
                self.root.after(0, lambda: self.upload_btn.configure(
                    text="📁 Upload PDF", state='normal'
                ))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"An error occurred: {str(e)}"
                ))
        
        threading.Thread(target=process_pdf, daemon=True).start()
    
    def send_message(self):
        """Send a message and get AI response."""
        message = self.message_entry.get('1.0', 'end-1c').strip()
        
        if not message or self.placeholder_active:
            return
        
        # Clear input
        self.message_entry.delete('1.0', 'end')
        self.add_placeholder()
        
        # Add user message
        self.add_message("user", message)
        
        # Update status
        self.update_status("processing")
        
        def get_response():
            try:
                if self.chat_mode == "pdf" and self.chatbot.pdf_loaded:
                    response, _ = self.chatbot.chat_with_pdf(message)
                else:
                    response = self.chatbot.chat_general(message)
                
                self.root.after(0, lambda: self.add_message("assistant", response))
                self.root.after(0, lambda: self.update_status("ready"))
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.root.after(0, lambda: self.add_message("assistant", error_msg))
                self.root.after(0, lambda: self.update_status("error"))
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def clear_chat(self):
        """Clear all chat messages."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.add_welcome_message()
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


def main():
    """Main function to run the Modern Chat GUI."""
    app = ModernChatGUI()
    app.run()


if __name__ == "__main__":
    main()
