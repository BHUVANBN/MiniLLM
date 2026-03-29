# MiniLLM - Offline AI Assistant (Ollama RAG)

MiniLLM is a powerful, fully offline AI assistant that allows you to chat with your PDF documents and folders using local Large Language Models via **Ollama**.

## 🌟 Features

- **100% Offline**: No data leaves your machine. No API keys required.
- **Local RAG Pipeline**: Powered by `LangChain`, `ChromaDB`, and `SentenceTransformers`.
- **Ollama Integration**: Use any local model (default: `llama3`, `mistral`, etc.).
- **Multi-Document Support**: Process single PDFs or entire directories.
- **Rich User Interfaces**:
  - **Desktop App**: Sleek Tkinter-based GUI with voice support.
  - **Web App**: Modern Gradio-based interface for browser access.
- **Voice Interactions**: Talk to your documents using built-in speech-to-text and text-to-speech.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download and install Ollama](https://ollama.com/)
- **Pull a model**: 
  ```bash
  ollama pull llama3
  ```

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Launch

**For Desktop GUI:**
```bash
python chatgpt_ui.py
```

**For Web Interface:**
```bash
python web_app.py
```

## 📂 Project Structure

- `universal_chatbot.py`: Core logic for RAG and Ollama integration.
- `chatgpt_ui.py`: Desktop application using Tkinter.
- `web_app.py`: Web interface using Gradio.
- `voice_chat.py`: Module for voice recognition and synthesis.
- `data/`: Sample documents and vector store storage.

## 🛠️ Configuration

You can configure the default model in `universal_chatbot.py`:
```python
self.ollama_model = "llama3"
```

## 📝 License
MIT License. Free to use and modify.
