# 🤖 MiniLLM Universal PDF Chatbot

A powerful AI chatbot that can chat with any PDF document or work as a general AI assistant using advanced Retrieval-Augmented Generation (RAG) technology.

## ✨ Features

- **📄 Any PDF Support**: Upload and chat with any PDF document
- **🤖 General AI Chat**: Use without PDF for general conversations
- **🖥️ Multiple Interfaces**: GUI (Tkinter), CLI, and Web (Gradio) options
- **🔍 Smart Search**: FAISS vector database for fast document search
- **🌐 Cross-Platform**: Runs on Windows, macOS, and Linux
- **🔒 Secure**: API keys handled securely with environment variables

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys (Free)

**HuggingFace Token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
**Cohere API Key**: [dashboard.cohere.ai/api-keys](https://dashboard.cohere.ai/api-keys)

Add them to `.env` file:
```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
COHERE_API_KEY=your_key_here
```

### 3. Run the Application

**Easy Launch (Recommended):**
```bash
# Linux/macOS
./run_minillm.sh

# Windows
run_minillm.bat

# Any OS
python launcher.py
```

**Direct Launch:**
```bash
python pdf_chat_gui.py    # GUI with PDF upload
python chat_cli.py        # CLI for general chat
```

## 💡 Usage Examples

### 🖥️ GUI Mode
1. Launch with `python launcher.py` → Choose option 1
2. Click "📁 Upload PDF" to select any document
3. Ask questions about the PDF or switch to general chat

### 💻 CLI Mode  
1. Launch with `python launcher.py` → Choose option 2
2. Chat directly with AI (no PDF needed)
3. Type `help` for commands, `quit` to exit

### 📄 PDF Questions
- "What is this document about?"
- "Summarize the main points"
- "Find information about [topic]"

### 🤖 General Questions
- "Explain quantum physics"
- "Write a poem about nature"
- "Help me plan a trip to Japan"

## 📁 Project Structure

```
├── run_minillm.sh          # Linux/macOS launcher
├── run_minillm.bat         # Windows launcher
├── launcher.py             # Universal launcher (recommended)
├── pdf_chat_gui.py         # GUI app for PDF chat
├── chat_cli.py             # CLI for general chat
├── universal_chatbot.py    # Core chatbot engine
├── requirements.txt        # Dependencies
├── .env                    # Your API keys
└── README.md              # This guide
```

## 🔧 Troubleshooting

**API Key Issues**: Verify keys are correct in `.env` file
**PDF Not Loading**: Check file path and permissions  
**Memory Issues**: Large PDFs need more RAM
**Network Issues**: Ensure stable internet for API calls

Check logs: `universal_chatbot.log`

## 📋 Requirements

- **OS**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 4GB (8GB recommended for large PDFs)
- **Internet**: Required for AI API calls

## 🤝 Contributing

Contributions welcome! Submit issues or pull requests.

---

**Note**: This is an AI assistant tool. For professional/legal documents, always consult qualified experts.
