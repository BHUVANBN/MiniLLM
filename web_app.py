#!/usr/bin/env python3
"""
MiniLLM Web Interface using Gradio

A simple web interface for the MiniLLM chatbot with PDF upload functionality.
"""

import gradio as gr
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_chatbot import UniversalChatbot

class WebApp:
    def __init__(self):
        self.chatbot = UniversalChatbot()
        self.chat_mode = "general"
        
        # Initialize chatbot
        success = self.chatbot.setup_local_models()
        if not success:
            print("❌ Failed to initialize local models. Ensure Ollama is running.")
    
    def upload_pdf(self, file_path):
        """Handle PDF or Directory upload."""
        if file_path is None:
            return "❌ No file/folder selected", "No content loaded"
        
        try:
            # Check if it's a directory or a file
            if os.path.isdir(file_path):
                success = self.chatbot.load_directory(file_path)
                prefix = "Folder"
            else:
                success = self.chatbot.process_pdf(file_path)
                prefix = "PDF"
            
            if success:
                # Need to trigger RAG index creation if not already part of load_directory/process_pdf
                # In current implementation, process_pdf does it all, but load_directory doesn't.
                if os.path.isdir(file_path):
                    self.chatbot.create_text_chunks()
                    self.chatbot.create_vector_database()
                
                name = Path(file_path).name
                return f"✅ Successfully loaded {prefix}: {name}", f"📄 {name}"
            else:
                return f"❌ Failed to process {prefix}", "No content loaded"
                
        except Exception as e:
            return f"❌ Error: {str(e)}", "No content loaded"
    
    def set_mode(self, mode):
        """Set chat mode."""
        if mode == "PDF Chat" and not self.chatbot.pdf_loaded:
            return "❌ Please upload a PDF first to use PDF chat mode"
        
        self.chat_mode = "pdf" if mode == "PDF Chat" else "general"
        return f"✅ Switched to {mode} mode"
    
    def chat(self, message, history):
        """Handle chat messages."""
        if not message.strip():
            return history, ""
        
        try:
            # Get AI response
            if self.chat_mode == "pdf" and self.chatbot.pdf_loaded:
                response, _ = self.chatbot.chat_with_pdf(message)
            else:
                response = self.chatbot.chat_general(message)
            
            # Add to history in new format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def clear_chat(self):
        """Clear chat history."""
        return []
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="MiniLLM AI Assistant",
            theme=gr.themes.Soft(primary_hue="purple"),
            css="""
            .gradio-container {
                max-width: 100% !important;
                margin: 0 !important;
                padding: 20px !important;
            }
            .contain {
                max-width: 100% !important;
            }
            .chatbot {
                height: 70vh !important;
            }
            .input-container {
                max-width: 100% !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 MiniLLM AI Assistant")
            gr.Markdown("Upload a PDF and chat with it, or have a general conversation!")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    # Mode selection
                    gr.Markdown("### 🎯 Chat Mode")
                    mode_radio = gr.Radio(
                        choices=["General Chat", "PDF Chat"],
                        value="General Chat",
                        label="",
                        show_label=False
                    )
                    
                    gr.Markdown("### 📄 Document Upload")
                    # PDF/Folder path upload
                    content_path = gr.Textbox(
                        label="File/Folder Path",
                        placeholder="Enter absolute path to PDF or Folder",
                        show_label=True
                    )
                    
                    # Upload button
                    upload_btn = gr.Button("🚀 Process Content", variant="secondary", size="lg")
                    
                    # PDF status
                    pdf_status = gr.Textbox(
                        label="Status",
                        value="No PDF loaded",
                        interactive=False,
                        lines=2
                    )
                    
                    gr.Markdown("### ⚡ Actions")
                    # Clear chat button
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="lg")
                
                with gr.Column(scale=3):
                    # Chat interface - much larger
                    chatbot_interface = gr.Chatbot(
                        label="",
                        height=600,
                        show_label=False,
                        elem_classes=["chatbot"],
                        container=True,
                        show_copy_button=True,
                        type="messages"
                    )
                    
                    # Input row
                    with gr.Row():
                        # Message input
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="Type your message here... (Press Enter to send)",
                            lines=2,
                            max_lines=5,
                            scale=4,
                            show_label=False,
                            container=False
                        )
                        
                        # Send button
                        send_btn = gr.Button("➤", variant="primary", size="lg", scale=1)
            
            # Status messages
            status_msg = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
            
            # Event handlers
            upload_btn.click(
                fn=self.upload_pdf,
                inputs=[content_path],
                outputs=[status_msg, pdf_status]
            )
            
            mode_radio.change(
                fn=self.set_mode,
                inputs=[mode_radio],
                outputs=[status_msg]
            )
            
            send_btn.click(
                fn=self.chat,
                inputs=[msg_input, chatbot_interface],
                outputs=[chatbot_interface, msg_input]
            )
            
            msg_input.submit(
                fn=self.chat,
                inputs=[msg_input, chatbot_interface],
                outputs=[chatbot_interface, msg_input]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot_interface]
            )
            
            # Welcome message
            interface.load(
                fn=lambda: [
                    {"role": "assistant", "content": "Hello! I'm your Offline AI assistant powered by Ollama. Upload a PDF or provide a folder path to chat about its content. How can I help you today?"}
                ],
                outputs=[chatbot_interface]
            )
        
        return interface


def main():
    """Main function to launch the web app."""
    try:
        print("🚀 Starting MiniLLM Web Interface...")
        
        app = WebApp()
        interface = app.create_interface()
        
        print("✅ Web interface ready!")
        print("🌐 Opening in your browser...")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            favicon_path=None
        )
        
    except ImportError as e:
        print(f"❌ Missing Gradio: {e}")
        print("📦 Install with: pip install gradio")
    except Exception as e:
        print(f"❌ Error launching web app: {e}")


if __name__ == "__main__":
    main()
