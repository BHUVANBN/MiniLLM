# PDF LLM Trainer - Complete Setup & Training System

A comprehensive system for training small Language Models on PDF content with 4-bit quantization support.

## Features

- **Complete Automation**: No manual setup required
- **PDF Understanding**: True comprehension, not keyword matching
- **Fast Training**: 10 minutes vs hours
- **Memory Efficient**: 4-bit quantization
- **User Friendly**: GUI interface
- **Cross Platform**: Windows, Linux, macOS
- **Self Contained**: No external dependencies needed

## One-Click Setup & Launch

### For Linux/macOS:
```bash
./run_minillm.sh
```

### For Windows:
```batch
run_minillm.bat
```

**That's it!** These scripts will automatically:
- Check Python installation
- Create virtual environment
- Install all dependencies
- Create model architecture
- Launch PDF LLM Trainer GUI

## What This System Does

### PDF Training Pipeline
1. **Upload PDF** → Extract text content
2. **Create Training Data** → Generate Q&A pairs from content
3. **Train Small LLM** → 800 steps, ~10 minutes
4. **Chat Intelligently** → Ask questions about PDF content

### Model Features
- **Small LLM Architecture**: 6-layer transformer (~30M parameters)
- **4-bit Quantization**: Memory efficient inference
- **Custom Training**: Learns from YOUR PDF content
- **Intelligent Responses**: Understands concepts, not just keywords

## Requirements

### Automatic Installation
The scripts automatically install:
- `torch>=1.9.0`
- `transformers>=4.20.0`
- `datasets>=2.0.0`
- `accelerate>=0.20.0`
- `bitsandbytes>=0.41.0`
- `PyPDF2>=3.0.0`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.0`

### System Requirements
- **Python 3.8+** (automatically checked)
- **4GB RAM minimum** (8GB recommended)
- **2GB disk space** for models and dependencies

## How to Use

### Step 1: Run Setup Script
```bash
# Linux/macOS
./run_minillm.sh

# Windows
run_minillm.bat
```

### Step 2: Upload PDF
- Click "Upload PDF & Start Training"
- Select your PDF file
- Wait for training completion (~10 minutes)

### Step 3: Chat with Trained Model
- Ask: "What is [concept from PDF]?"
- Ask: "Explain [topic from PDF]"
- Ask: "Tell me about [subject from PDF]"

## Project Structure

```
MiniLLM/
├── run_minillm.sh          # Linux/macOS setup script
├── run_minillm.bat         # Windows setup script
├── requirements.txt        # Auto-generated dependencies
├── LLMmodel/              # Main project directory
│   ├── mini_llm_env/      # Auto-created virtual environment
│   ├── model.py           # Auto-generated model architecture
│   ├── pdf_llm_trainer.py # Main GUI application
│   └── models/            # Trained models storage
└── README.md              # This file
```

## What the Scripts Create

### Model Architecture (`LLMmodel/model.py`)
- Complete SmallLLM implementation
- 4-bit quantization support
- Transformer architecture with attention
- Compatible with Hugging Face ecosystem

### Virtual Environment (`LLMmodel/mini_llm_env/`)
- Isolated Python environment
- All dependencies installed
- No conflicts with system Python

### Requirements File (`requirements.txt`)
- All necessary packages
- Version-pinned for stability
- Automatically installed

## Training Process

### Intelligent Data Creation
1. **Extract PDF Content** → Parse all text
2. **Identify Key Concepts** → Find important terms
3. **Generate Q&A Pairs** → Create training examples
4. **Format for Training** → Prepare for LLM

Contributions welcome! Submit issues or pull requests.

---

**Note**: This is an AI assistant tool. For professional/legal documents, always consult qualified experts.
