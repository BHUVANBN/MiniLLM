#!/bin/bash

echo "🤖 PDF LLM Trainer - Complete Setup & Launch (Linux/macOS)"
echo "=========================================================="
echo "This script will set up everything needed for PDF LLM training"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "LLMmodel/mini_llm_env" ]; then
    echo "📦 Creating virtual environment..."
    cd LLMmodel
    python3 -m venv mini_llm_env
    cd ..
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source LLMmodel/mini_llm_env/bin/activate

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "📝 Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
PyPDF2>=3.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
cohere>=5.0.0
openai>=1.0.0
huggingface-hub>=0.16.0
python-dotenv>=1.0.0
EOF
fi

# Install requirements
echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create the model architecture if it doesn't exist
if [ ! -f "LLMmodel/model.py" ]; then
    echo "🧠 Creating model architecture..."
    cat > LLMmodel/model.py << 'EOF'
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import math

class SmallLLMConfig(PretrainedConfig):
    model_type = "small_llm"
    
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        super().__init__(**kwargs)

class SmallLLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)))
        
    def forward(self, hidden_states, attention_mask=None):
        B, T, C = hidden_states.size()
        
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class SmallLLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = SmallLLMAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states

class SmallLLM(PreTrainedModel):
    config_class = SmallLLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([SmallLLMBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.init_weights()
        
    def get_input_embeddings(self):
        return self.wte
        
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify input_ids")
            
        device = input_ids.device
        seq_length = input_ids.shape[1]
        
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)
            
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
        )
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            total_params = 0
            for name, param in self.named_parameters():
                if 'embed' not in name.lower():
                    total_params += param.numel()
            return total_params
        else:
            return sum(p.numel() for p in self.parameters())

def load_pretrained_model(model_path, quantization_config=None):
    from transformers import GPT2Tokenizer
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        if quantization_config:
            from transformers import BitsAndBytesConfig
            model = SmallLLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            model = SmallLLM.from_pretrained(model_path)
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        config = SmallLLMConfig()
        model = SmallLLM(config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
EOF
fi

# Create the main PDF LLM trainer if it doesn't exist
if [ ! -f "LLMmodel/pdf_llm_trainer.py" ]; then
    echo "🎯 Creating PDF LLM trainer..."
    cp LLMmodel/pdf_llm_trainer.py LLMmodel/pdf_llm_trainer_backup.py 2>/dev/null || true
fi

# API Key Collection and Verification
echo "🔑 API Key Setup"
echo "==============="
echo "This application supports optional API integrations for enhanced features."
echo

# Create .env file for API keys
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "# API Keys for MiniLLM Project" > $ENV_FILE
    echo "# Generated on $(date)" >> $ENV_FILE
    echo "" >> $ENV_FILE
fi

# Function to verify Cohere API key
verify_cohere_key() {
    local api_key=$1
    echo "🔍 Verifying Cohere API key..."
    
    # Test API key with a simple request
    python3 -c "
import cohere
import sys
try:
    co = cohere.Client('$api_key')
    response = co.generate(
        model='command-light',
        prompt='Hello',
        max_tokens=5
    )
    print('✅ Cohere API key verified successfully!')
    sys.exit(0)
except Exception as e:
    print('❌ Cohere API key verification failed:', str(e))
    sys.exit(1)
" 2>/dev/null
    
    return $?
}

# Collect Cohere API Key
echo "📡 Cohere API Integration (Optional)"
echo "Cohere provides advanced language model capabilities."
echo "Get your free API key from: https://dashboard.cohere.ai/api-keys"
echo
read -p "Enter your Cohere API key (or press Enter to skip): " cohere_key

if [ ! -z "$cohere_key" ]; then
    if verify_cohere_key "$cohere_key"; then
        echo "COHERE_API_KEY=$cohere_key" >> $ENV_FILE
        echo "✅ Cohere API key saved successfully!"
    else
        echo "⚠️  Invalid Cohere API key. Continuing without Cohere integration."
        echo "You can add it later by editing the .env file."
    fi
else
    echo "⏭️  Skipping Cohere API setup."
fi

echo

# Function to verify OpenAI API key (if needed)
verify_openai_key() {
    local api_key=$1
    echo "🔍 Verifying OpenAI API key..."
    
    python3 -c "
import openai
import sys
try:
    openai.api_key = '$api_key'
    response = openai.Model.list()
    print('✅ OpenAI API key verified successfully!')
    sys.exit(0)
except Exception as e:
    print('❌ OpenAI API key verification failed:', str(e))
    sys.exit(1)
" 2>/dev/null
    
    return $?
}

# Collect OpenAI API Key (Optional)
echo "🤖 OpenAI API Integration (Optional)"
echo "OpenAI provides GPT models for enhanced capabilities."
echo "Get your API key from: https://platform.openai.com/api-keys"
echo
read -p "Enter your OpenAI API key (or press Enter to skip): " openai_key

if [ ! -z "$openai_key" ]; then
    # Check if openai package is available
    python3 -c "import openai" 2>/dev/null
    if [ $? -eq 0 ]; then
        if verify_openai_key "$openai_key"; then
            echo "OPENAI_API_KEY=$openai_key" >> $ENV_FILE
            echo "✅ OpenAI API key saved successfully!"
        else
            echo "⚠️  Invalid OpenAI API key. Continuing without OpenAI integration."
        fi
    else
        echo "OPENAI_API_KEY=$openai_key" >> $ENV_FILE
        echo "⚠️  OpenAI package not installed. API key saved for future use."
    fi
else
    echo "⏭️  Skipping OpenAI API setup."
fi

echo

# Collect Hugging Face Token (Optional)
echo "🤗 Hugging Face Hub Integration (Optional)"
echo "Hugging Face provides access to thousands of pre-trained models."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo
read -p "Enter your Hugging Face token (or press Enter to skip): " hf_token

if [ ! -z "$hf_token" ]; then
    echo "HUGGINGFACE_HUB_TOKEN=$hf_token" >> $ENV_FILE
    echo "✅ Hugging Face token saved successfully!"
    
    # Set environment variable for current session
    export HUGGINGFACE_HUB_TOKEN=$hf_token
else
    echo "⏭️  Skipping Hugging Face setup."
fi

echo

# Summary of API setup
echo "📋 API Setup Summary"
echo "==================="
if [ -f "$ENV_FILE" ] && [ -s "$ENV_FILE" ]; then
    echo "✅ API keys saved to .env file"
    echo "📁 Location: $(pwd)/.env"
    echo "🔒 Keep this file secure and don't share it publicly!"
    echo
    echo "Configured APIs:"
    if grep -q "COHERE_API_KEY" $ENV_FILE; then
        echo "  ✅ Cohere API - Enhanced language generation"
    fi
    if grep -q "OPENAI_API_KEY" $ENV_FILE; then
        echo "  ✅ OpenAI API - GPT model access"
    fi
    if grep -q "HUGGINGFACE_HUB_TOKEN" $ENV_FILE; then
        echo "  ✅ Hugging Face Hub - Model repository access"
    fi
else
    echo "⏭️  No API keys configured. Using offline-only features."
fi

echo
echo "✅ Setup complete!"
echo "🚀 Starting PDF LLM Trainer..."
echo

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(cat $ENV_FILE | grep -v '^#' | xargs)
fi

# Change to LLMmodel directory and run
cd LLMmodel
python pdf_llm_trainer.py
