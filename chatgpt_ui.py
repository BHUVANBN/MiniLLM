#!/usr/bin/env python3
"""
MiniLLM ChatGPT-style UI - Clean and Modern Interface

A ChatGPT-inspired interface using pure tkinter with clean design.
No borders, modern layout, and smooth user experience.
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
from voice_chat import VoiceChat

class ChatGPTUI:
    def __init__(self):
        self.root = tk.Tk()
        self.chatbot = UniversalChatbot()
        self.chat_mode = "general"  # "general" or "pdf"
        self.current_pdf = None
        
        # Initialize voice chat
        self.voice_chat = VoiceChat()
        self.voice_enabled = False
        
        self.setup_window()
        self.create_widgets()
        
        # Setup voice chat callbacks
        if self.voice_chat.is_available():
            self.setup_voice_callbacks()
        
        # Initialize chatbot in background
        threading.Thread(target=self.initialize_chatbot, daemon=True).start()
    
    def setup_window(self):
        """Configure the main window."""
        self.root.title("MiniLLM")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.root.winfo_screenheight() // 2) - (750 // 2)
        self.root.geometry(f"1000x750+{x}+{y}")
        
        # ChatGPT-like color scheme
        self.colors = {
            'bg': '#212121',           # Dark background
            'sidebar': '#171717',      # Darker sidebar
            'chat_bg': '#212121',      # Chat background
            'user_bubble': '#2f2f2f',  # User message background
            'ai_bubble': '#212121',    # AI message background (same as chat)
            'text': '#ececec',         # Primary text
            'text_dim': '#9ca3af',     # Dimmed text
            'input_bg': '#2f2f2f',     # Input background
            'button': '#10a37f',       # Green button (ChatGPT style)
            'button_hover': '#0d8a6b', # Button hover
            'border': '#4d4d4f',       # Subtle borders
        }
        
        self.root.configure(bg=self.colors['bg'])
    
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True)
        
        # Left sidebar
        self.create_sidebar(main_container)
        
        # Main chat area
        self.create_chat_area(main_container)
    
    def create_sidebar(self, parent):
        """Create the left sidebar."""
        sidebar = tk.Frame(parent, bg=self.colors['sidebar'], width=260)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        
        # Sidebar content
        sidebar_content = tk.Frame(sidebar, bg=self.colors['sidebar'])
        sidebar_content.pack(fill='both', expand=True, padx=12, pady=16)
        
        # New Chat button
        new_chat_btn = tk.Button(
            sidebar_content,
            text="+ New chat",
            bg=self.colors['sidebar'],
            fg=self.colors['text'],
            font=('Segoe UI', 11),
            relief='flat',
            bd=0,
            padx=12,
            pady=8,
            anchor='w',
            cursor='hand2',
            command=self.new_chat
        )
        new_chat_btn.pack(fill='x', pady=(0, 16))
        
        # Mode selection
        mode_label = tk.Label(
            sidebar_content,
            text="Mode",
            bg=self.colors['sidebar'],
            fg=self.colors['text_dim'],
            font=('Segoe UI', 10),
            anchor='w'
        )
        mode_label.pack(fill='x', pady=(0, 8))
        
        self.mode_var = tk.StringVar(value="general")
        
        # General mode button
        self.general_btn = tk.Button(
            sidebar_content,
            text="💬 General Chat",
            bg=self.colors['input_bg'] if self.mode_var.get() == "general" else self.colors['sidebar'],
            fg=self.colors['text'],
            font=('Segoe UI', 10),
            relief='flat',
            bd=0,
            padx=12,
            pady=8,
            anchor='w',
            cursor='hand2',
            command=lambda: self.set_mode("general")
        )
        self.general_btn.pack(fill='x', pady=(0, 4))
        
        # PDF mode button
        self.pdf_btn = tk.Button(
            sidebar_content,
            text="📄 PDF Chat",
            bg=self.colors['sidebar'],
            fg=self.colors['text'],
            font=('Segoe UI', 10),
            relief='flat',
            bd=0,
            padx=12,
            pady=8,
            anchor='w',
            cursor='hand2',
            command=lambda: self.set_mode("pdf")
        )
        self.pdf_btn.pack(fill='x', pady=(0, 16))
        
        # PDF section
        pdf_label = tk.Label(
            sidebar_content,
            text="Documents",
            bg=self.colors['sidebar'],
            fg=self.colors['text_dim'],
            font=('Segoe UI', 10),
            anchor='w'
        )
        pdf_label.pack(fill='x', pady=(0, 8))
        
        # Upload PDF button
        self.upload_btn = tk.Button(
            sidebar_content,
            text="📁 Upload PDF",
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            font=('Segoe UI', 10),
            relief='flat',
            bd=0,
            padx=12,
            pady=8,
            cursor='hand2',
            command=self.upload_pdf
        )
        self.upload_btn.pack(fill='x', pady=(0, 8))
        
        # PDF status
        self.pdf_status = tk.Label(
            sidebar_content,
            text="No document loaded",
            bg=self.colors['sidebar'],
            fg=self.colors['text_dim'],
            font=('Segoe UI', 9),
            anchor='w',
            wraplength=220
        )
        self.pdf_status.pack(fill='x')
        
        # Voice Chat section
        if self.voice_chat.is_available():
            voice_label = tk.Label(
                sidebar_content,
                text="Voice Chat",
                bg=self.colors['sidebar'],
                fg=self.colors['text_dim'],
                font=('Segoe UI', 10),
                anchor='w'
            )
            voice_label.pack(fill='x', pady=(16, 8))
            
            # Voice toggle
            self.voice_toggle_btn = tk.Button(
                sidebar_content,
                text="🎤 Enable Voice",
                bg=self.colors['input_bg'],
                fg=self.colors['text'],
                font=('Segoe UI', 10),
                relief='flat',
                bd=0,
                padx=12,
                pady=8,
                cursor='hand2',
                command=self.toggle_voice_chat
            )
            self.voice_toggle_btn.pack(fill='x', pady=(0, 8))
            
            # Voice controls frame
            self.voice_controls = tk.Frame(sidebar_content, bg=self.colors['sidebar'])
            self.voice_controls.pack(fill='x', pady=(0, 8))
            
            # Listen button
            self.listen_btn = tk.Button(
                self.voice_controls,
                text="🎙️ Listen",
                bg=self.colors['input_bg'],
                fg=self.colors['text'],
                font=('Segoe UI', 9),
                relief='flat',
                bd=0,
                padx=8,
                pady=6,
                cursor='hand2',
                command=self.start_voice_input,
                state='disabled'
            )
            self.listen_btn.pack(side='left', padx=(0, 4))
            
            # Stop button
            self.stop_btn = tk.Button(
                self.voice_controls,
                text="⏹️ Stop",
                bg=self.colors['input_bg'],
                fg=self.colors['text'],
                font=('Segoe UI', 9),
                relief='flat',
                bd=0,
                padx=8,
                pady=6,
                cursor='hand2',
                command=self.stop_voice,
                state='disabled'
            )
            self.stop_btn.pack(side='left')
            
            # Voice status
            self.voice_status = tk.Label(
                sidebar_content,
                text="Voice chat disabled",
                bg=self.colors['sidebar'],
                fg=self.colors['text_dim'],
                font=('Segoe UI', 8),
                anchor='w'
            )
            self.voice_status.pack(fill='x', pady=(0, 8))
        
        # Status at bottom
        status_container = tk.Frame(sidebar_content, bg=self.colors['sidebar'])
        status_container.pack(side='bottom', fill='x', pady=(20, 0))
        
        self.status_label = tk.Label(
            status_container,
            text="Initializing...",
            bg=self.colors['sidebar'],
            fg=self.colors['text_dim'],
            font=('Segoe UI', 9),
            anchor='w'
        )
        self.status_label.pack(fill='x')
    
    def create_chat_area(self, parent):
        """Create the main chat area."""
        chat_container = tk.Frame(parent, bg=self.colors['chat_bg'])
        chat_container.pack(side='right', fill='both', expand=True)
        
        # Chat messages area
        self.create_messages_area(chat_container)
        
        # Input area
        self.create_input_area(chat_container)
    
    def create_messages_area(self, parent):
        """Create the scrollable messages area."""
        # Messages container
        messages_container = tk.Frame(parent, bg=self.colors['chat_bg'])
        messages_container.pack(fill='both', expand=True)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            messages_container,
            bg=self.colors['chat_bg'],
            highlightthickness=0,
            bd=0
        )
        
        # Scrollbar
        scrollbar = tk.Scrollbar(
            messages_container,
            orient="vertical",
            command=self.canvas.yview,
            bg=self.colors['chat_bg'],
            troughcolor=self.colors['chat_bg'],
            bd=0,
            highlightthickness=0
        )
        
        # Scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['chat_bg'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to multiple widgets for better scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)
        messages_container.bind("<MouseWheel>", self._on_mousewheel)
        
        # Bind mousewheel to the main window for global scrolling
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        # Add keyboard scrolling support
        self.root.bind("<Prior>", self._on_page_up)      # Page Up
        self.root.bind("<Next>", self._on_page_down)     # Page Down
        self.root.bind("<Up>", self._on_arrow_up)        # Arrow Up
        self.root.bind("<Down>", self._on_arrow_down)    # Arrow Down
        self.root.bind("<Home>", self._on_home)          # Home
        self.root.bind("<End>", self._on_end)            # End
        
        # Also bind Button-4 and Button-5 for Linux systems
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)
        self.scrollable_frame.bind("<Button-4>", self._on_mousewheel_linux)
        self.scrollable_frame.bind("<Button-5>", self._on_mousewheel_linux)
        messages_container.bind("<Button-4>", self._on_mousewheel_linux)
        messages_container.bind("<Button-5>", self._on_mousewheel_linux)
        self.root.bind("<Button-4>", self._on_mousewheel_linux)
        self.root.bind("<Button-5>", self._on_mousewheel_linux)
        
        # Welcome message
        self.add_welcome_message()
    
    def create_input_area(self, parent):
        """Create the input area at the bottom."""
        input_container = tk.Frame(parent, bg=self.colors['chat_bg'])
        input_container.pack(fill='x', side='bottom')
        
        # Input wrapper for centering
        input_wrapper = tk.Frame(input_container, bg=self.colors['chat_bg'])
        input_wrapper.pack(pady=20)
        
        # Input frame
        input_frame = tk.Frame(
            input_wrapper,
            bg=self.colors['input_bg'],
            highlightthickness=0
        )
        input_frame.pack(padx=20)
        
        # Text input
        self.message_entry = tk.Text(
            input_frame,
            height=1,
            wrap='word',
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            font=('Segoe UI', 12),
            relief='flat',
            bd=0,
            padx=16,
            pady=12,
            insertbackground=self.colors['text'],
            selectbackground=self.colors['button'],
            selectforeground='white',
            width=70
        )
        self.message_entry.pack(side='left', fill='both')
        
        # Voice chat button (if available)
        if self.voice_chat.is_available():
            self.voice_btn = tk.Button(
                input_frame,
                text="🎤",
                bg='#03dac6',  # Teal color
                fg='white',
                font=('Segoe UI', 12, 'bold'),
                relief='flat',
                bd=0,
                width=3,
                cursor='hand2',
                command=self.toggle_voice_input
            )
            self.voice_btn.pack(side='right', padx=(4, 4), pady=8)
        
        # Send button
        self.send_btn = tk.Button(
            input_frame,
            text="➤",
            bg=self.colors['button'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief='flat',
            bd=0,
            width=3,
            cursor='hand2',
            command=self.send_message
        )
        self.send_btn.pack(side='right', padx=(8, 8), pady=8)
        
        # Bind events
        self.message_entry.bind('<Return>', self.on_enter_key)
        self.message_entry.bind('<KeyRelease>', self.on_key_release)
        
        # Button hover effects
        self.send_btn.bind('<Enter>', lambda e: self.send_btn.configure(bg=self.colors['button_hover']))
        self.send_btn.bind('<Leave>', lambda e: self.send_btn.configure(bg=self.colors['button']))
    
    def add_welcome_message(self):
        """Add welcome message to chat."""
        welcome_text = """Hello! I'm your AI assistant. I can help you with:

• General questions and conversations
• PDF document analysis and Q&A
• Information lookup and explanations
• Creative writing and brainstorming

How can I assist you today?"""
        
        self.add_message("assistant", welcome_text)
    
    def add_message(self, sender, message, timestamp=None):
        """Add a message to the chat."""
        import datetime
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%H:%M")
        
        # Message container
        msg_container = tk.Frame(self.scrollable_frame, bg=self.colors['chat_bg'])
        msg_container.pack(fill='x', padx=20, pady=10)
        
        if sender == "user":
            # User message (right aligned)
            msg_frame = tk.Frame(msg_container, bg=self.colors['chat_bg'])
            msg_frame.pack(anchor='e')
            
            # Message bubble
            bubble = tk.Frame(msg_frame, bg=self.colors['user_bubble'])
            bubble.pack(anchor='e')
            
            # Message text
            msg_label = tk.Label(
                bubble,
                text=message,
                bg=self.colors['user_bubble'],
                fg=self.colors['text'],
                font=('Segoe UI', 11),
                wraplength=500,
                justify='left',
                padx=16,
                pady=12
            )
            msg_label.pack()
            
        else:
            # Assistant message (left aligned)
            msg_frame = tk.Frame(msg_container, bg=self.colors['chat_bg'])
            msg_frame.pack(anchor='w')
            
            # Avatar and message container
            content_frame = tk.Frame(msg_frame, bg=self.colors['chat_bg'])
            content_frame.pack(anchor='w')
            
            # AI Avatar
            avatar = tk.Label(
                content_frame,
                text="🤖",
                bg=self.colors['chat_bg'],
                font=('Segoe UI', 16)
            )
            avatar.pack(side='left', anchor='n', padx=(0, 12), pady=(4, 0))
            
            # Message text (no bubble for AI, like ChatGPT)
            msg_label = tk.Label(
                content_frame,
                text=message,
                bg=self.colors['chat_bg'],
                fg=self.colors['text'],
                font=('Segoe UI', 11),
                wraplength=600,
                justify='left',
                anchor='nw'
            )
            msg_label.pack(side='left', anchor='nw')
        
        # Auto-scroll to bottom
        self.root.after(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom."""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for Windows and macOS."""
        try:
            # Check if canvas exists and has content to scroll
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                # Windows and macOS
                if event.delta:
                    self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                # Alternative for some systems
                elif event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")
        except Exception:
            pass  # Ignore errors if canvas is not ready
    
    def _on_mousewheel_linux(self, event):
        """Handle mouse wheel scrolling for Linux systems."""
        try:
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                if event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")
        except Exception:
            pass  # Ignore errors if canvas is not ready
    
    def _on_page_up(self, event):
        """Handle Page Up key for scrolling."""
        try:
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                self.canvas.yview_scroll(-5, "units")
        except Exception:
            pass
    
    def _on_page_down(self, event):
        """Handle Page Down key for scrolling."""
        try:
            if hasattr(self, 'canvas') and self.canvas.winfo_exists():
                self.canvas.yview_scroll(5, "units")
        except Exception:
            pass
    
    def _on_arrow_up(self, event):
        """Handle Arrow Up key for scrolling."""
        try:
            # Only scroll if not focused on text input
            if hasattr(self, 'canvas') and self.canvas.winfo_exists() and self.root.focus_get() != self.message_entry:
                self.canvas.yview_scroll(-1, "units")
                return "break"
        except Exception:
            pass
    
    def _on_arrow_down(self, event):
        """Handle Arrow Down key for scrolling."""
        try:
            # Only scroll if not focused on text input
            if hasattr(self, 'canvas') and self.canvas.winfo_exists() and self.root.focus_get() != self.message_entry:
                self.canvas.yview_scroll(1, "units")
                return "break"
        except Exception:
            pass
    
    def _on_home(self, event):
        """Handle Home key to scroll to top."""
        try:
            if hasattr(self, 'canvas') and self.canvas.winfo_exists() and self.root.focus_get() != self.message_entry:
                self.canvas.yview_moveto(0.0)
                return "break"
        except Exception:
            pass
    
    def _on_end(self, event):
        """Handle End key to scroll to bottom."""
        try:
            if hasattr(self, 'canvas') and self.canvas.winfo_exists() and self.root.focus_get() != self.message_entry:
                self.canvas.yview_moveto(1.0)
                return "break"
        except Exception:
            pass
    
    def on_enter_key(self, event):
        """Handle Enter key press."""
        if event.state & 0x4:  # Ctrl+Enter for new line
            return
        else:  # Enter to send
            self.send_message()
            return "break"
    
    def on_key_release(self, event):
        """Handle key release for auto-resize."""
        # Auto-resize text input
        content = self.message_entry.get("1.0", "end-1c")
        lines = content.count('\n') + 1
        self.message_entry.configure(height=min(lines, 5))
    
    def set_mode(self, mode):
        """Set chat mode."""
        if mode == "pdf" and not self.chatbot.pdf_loaded:
            messagebox.showinfo("PDF Required", "Please upload a PDF document first.")
            return
        
        self.chat_mode = mode
        self.mode_var.set(mode)
        
        # Update button styles
        if mode == "general":
            self.general_btn.configure(bg=self.colors['input_bg'])
            self.pdf_btn.configure(bg=self.colors['sidebar'])
        else:
            self.general_btn.configure(bg=self.colors['sidebar'])
            self.pdf_btn.configure(bg=self.colors['input_bg'])
    
    def new_chat(self):
        """Start a new chat."""
        # Clear messages
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add welcome message
        self.add_welcome_message()
    
    def upload_pdf(self):
        """Handle PDF upload."""
        file_path = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not file_path:
            return
        
        self.status_label.configure(text="Processing PDF...")
        self.upload_btn.configure(text="Processing...", state='disabled')
        
        def process_pdf():
            try:
                success = self.chatbot.process_pdf(file_path)
                
                if success:
                    self.current_pdf = file_path
                    pdf_name = Path(file_path).name
                    
                    self.root.after(0, lambda: self.pdf_status.configure(
                        text=f"✓ {pdf_name}"
                    ))
                    self.root.after(0, lambda: self.status_label.configure(text="Ready"))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="📁 Upload PDF", state='normal'
                    ))
                    
                    # Add success message
                    self.root.after(0, lambda: self.add_message(
                        "assistant",
                        f"Successfully loaded {pdf_name}. You can now switch to PDF Chat mode to ask questions about the document."
                    ))
                    
                else:
                    self.root.after(0, lambda: self.status_label.configure(text="Error"))
                    self.root.after(0, lambda: self.upload_btn.configure(
                        text="📁 Upload PDF", state='normal'
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(text="Error"))
                self.root.after(0, lambda: self.upload_btn.configure(
                    text="📁 Upload PDF", state='normal'
                ))
        
        threading.Thread(target=process_pdf, daemon=True).start()
    
    def send_message(self):
        """Send message and get response."""
        message = self.message_entry.get("1.0", "end-1c").strip()
        
        if not message:
            return
        
        # Clear input
        self.message_entry.delete("1.0", "end")
        self.message_entry.configure(height=1)
        
        # Add user message
        self.add_message("user", message)
        
        # Show typing indicator
        self.status_label.configure(text="Thinking...")
        
        def get_response():
            try:
                if self.chat_mode == "pdf" and self.chatbot.pdf_loaded:
                    response, _ = self.chatbot.chat_with_pdf(message)
                else:
                    response = self.chatbot.chat_general(message)
                
                self.root.after(0, lambda: self.add_message("assistant", response))
                self.root.after(0, lambda: self.status_label.configure(text="Ready"))
                
                # Speak the response if voice is enabled
                if self.voice_enabled and self.voice_chat.is_available():
                    self.voice_chat.speak(response)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.root.after(0, lambda: self.add_message("assistant", error_msg))
                self.root.after(0, lambda: self.status_label.configure(text="Error"))
                
                # Speak error message if voice is enabled
                if self.voice_enabled and self.voice_chat.is_available():
                    self.voice_chat.speak(error_msg)
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def setup_voice_callbacks(self):
        """Setup voice chat event callbacks."""
        self.voice_chat.set_callbacks(
            on_speech_recognized=self.on_speech_recognized,
            on_listening_start=self.on_listening_start,
            on_listening_stop=self.on_listening_stop,
            on_speaking_start=self.on_speaking_start,
            on_speaking_stop=self.on_speaking_stop,
            on_error=self.on_voice_error
        )
    
    def toggle_voice_chat(self):
        """Toggle voice chat on/off."""
        if not self.voice_chat.is_available():
            messagebox.showerror("Voice Chat", "Voice chat is not available. Please install required dependencies.")
            return
        
        self.voice_enabled = not self.voice_enabled
        
        if self.voice_enabled:
            self.voice_toggle_btn.configure(text="🔇 Disable Voice")
            self.listen_btn.configure(state='normal')
            self.stop_btn.configure(state='normal')
            self.voice_status.configure(text="Voice chat enabled")
            # Update main voice button if it exists
            if hasattr(self, 'voice_btn'):
                self.voice_btn.configure(bg='#03dac6')
        else:
            self.voice_toggle_btn.configure(text="🎤 Enable Voice")
            self.listen_btn.configure(state='disabled')
            self.stop_btn.configure(state='disabled')
            self.voice_status.configure(text="Voice chat disabled")
            # Update main voice button if it exists
            if hasattr(self, 'voice_btn'):
                self.voice_btn.configure(bg='#555555')
            # Stop any ongoing voice operations
            self.voice_chat.stop_listening()
            self.voice_chat.stop_speaking()
    
    def toggle_voice_input(self):
        """Toggle voice input - main voice button functionality."""
        if not self.voice_chat.is_available():
            messagebox.showinfo("Voice Chat", "Voice chat is not available.\n\nPlease install dependencies:\nsudo apt-get install python3-pyaudio\npip install speechrecognition pyttsx3")
            return
        
        # Auto-enable voice chat if not enabled
        if not self.voice_enabled:
            self.voice_enabled = True
            if hasattr(self, 'voice_toggle_btn'):
                self.voice_toggle_btn.configure(text="🔇 Disable Voice")
                self.listen_btn.configure(state='normal')
                self.stop_btn.configure(state='normal')
                self.voice_status.configure(text="Voice chat enabled")
        
        # Toggle listening state
        if self.voice_chat.is_listening():
            self.stop_voice()
        else:
            self.start_voice_input()
    
    def start_voice_input(self):
        """Start listening for voice input."""
        if not self.voice_enabled:
            return
        
        if self.voice_chat.is_listening():
            return
        
        # Stop any ongoing speech
        self.voice_chat.stop_speaking()
        
        # Start listening
        success = self.voice_chat.start_listening()
        if not success:
            self.voice_status.configure(text="Failed to start listening")
    
    def stop_voice(self):
        """Stop all voice operations."""
        self.voice_chat.stop_listening()
        self.voice_chat.stop_speaking()
    
    def on_speech_recognized(self, text):
        """Handle recognized speech."""
        # Add the recognized text to the input field
        self.message_entry.delete("1.0", "end")
        self.message_entry.insert("1.0", text)
        
        # Automatically send the message
        self.send_message()
    
    def on_listening_start(self):
        """Handle listening start event."""
        self.root.after(0, lambda: self.voice_status.configure(text="🎙️ Listening..."))
        if hasattr(self, 'listen_btn'):
            self.root.after(0, lambda: self.listen_btn.configure(text="🎙️ Listening...", state='disabled'))
        if hasattr(self, 'voice_btn'):
            self.root.after(0, lambda: self.voice_btn.configure(text="🔴", bg='#2196f3'))
    
    def on_listening_stop(self):
        """Handle listening stop event."""
        self.root.after(0, lambda: self.voice_status.configure(text="Processing speech..."))
        if hasattr(self, 'listen_btn'):
            self.root.after(0, lambda: self.listen_btn.configure(text="🎙️ Listen", state='normal'))
        if hasattr(self, 'voice_btn'):
            self.root.after(0, lambda: self.voice_btn.configure(text="🎤", bg='#03dac6'))
    
    def on_speaking_start(self):
        """Handle speaking start event."""
        self.root.after(0, lambda: self.voice_status.configure(text="🔊 Speaking..."))
        if hasattr(self, 'stop_btn'):
            self.root.after(0, lambda: self.stop_btn.configure(text="⏹️ Stop Speaking"))
        if hasattr(self, 'voice_btn'):
            self.root.after(0, lambda: self.voice_btn.configure(text="🔊", bg='#4caf50'))
    
    def on_speaking_stop(self):
        """Handle speaking stop event."""
        self.root.after(0, lambda: self.voice_status.configure(text="Voice chat ready"))
        if hasattr(self, 'stop_btn'):
            self.root.after(0, lambda: self.stop_btn.configure(text="⏹️ Stop"))
        if hasattr(self, 'voice_btn'):
            self.root.after(0, lambda: self.voice_btn.configure(text="🎤", bg='#03dac6'))
    
    def on_voice_error(self, error_message):
        """Handle voice chat errors."""
        self.root.after(0, lambda: self.voice_status.configure(text=f"Error: {error_message}"))
        self.root.after(0, lambda: self.listen_btn.configure(text="🎙️ Listen", state='normal'))
    
    def initialize_chatbot(self):
        """Initialize chatbot in background."""
        try:
            success = self.chatbot.setup_api_keys()
            if success:
                self.root.after(0, lambda: self.status_label.configure(text="Ready"))
            else:
                self.root.after(0, lambda: self.status_label.configure(text="Error - Check API keys"))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(text="Error"))
    
    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nGoodbye!")


def main():
    """Main function."""
    app = ChatGPTUI()
    app.run()


if __name__ == "__main__":
    main()
