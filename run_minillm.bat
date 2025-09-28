@echo off
echo 🤖 PDF LLM Trainer - Complete Setup ^& Launch (Windows)
echo ======================================================
echo This script will set up everything needed for PDF LLM training
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist "LLMmodel\mini_llm_env" (
    echo 📦 Creating virtual environment...
    cd LLMmodel
    python -m venv mini_llm_env
    cd ..
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call LLMmodel\mini_llm_env\Scripts\activate.bat

REM Create requirements.txt if it doesn't exist
if not exist "requirements.txt" (
    echo 📝 Creating requirements.txt...
    (
        echo torch^>=1.9.0
        echo transformers^>=4.20.0
        echo datasets^>=2.0.0
        echo accelerate^>=0.20.0
        echo bitsandbytes^>=0.41.0
        echo PyPDF2^>=3.0.0
        echo numpy^>=1.21.0
        echo scikit-learn^>=1.0.0
        echo cohere^>=5.0.0
        echo openai^>=1.0.0
        echo huggingface-hub^>=0.16.0
        echo python-dotenv^>=1.0.0
    ) > requirements.txt
)

REM Install requirements
echo 📦 Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create the model architecture if it doesn't exist
if not exist "LLMmodel\model.py" (
    echo 🧠 Creating model architecture...
    (
        echo import torch
        echo import torch.nn as nn
        echo from transformers import PreTrainedModel, PretrainedConfig
        echo from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        echo import math
        echo.
        echo class SmallLLMConfig^(PretrainedConfig^):
        echo     model_type = "small_llm"
        echo     
        echo     def __init__^(
        echo         self,
        echo         vocab_size=50257,
        echo         n_positions=1024,
        echo         n_embd=768,
        echo         n_layer=12,
        echo         n_head=12,
        echo         n_inner=None,
        echo         activation_function="gelu_new",
        echo         resid_pdrop=0.1,
        echo         embd_pdrop=0.1,
        echo         attn_pdrop=0.1,
        echo         layer_norm_epsilon=1e-5,
        echo         initializer_range=0.02,
        echo         **kwargs
        echo     ^):
        echo         self.vocab_size = vocab_size
        echo         self.n_positions = n_positions
        echo         self.n_embd = n_embd
        echo         self.n_layer = n_layer
        echo         self.n_head = n_head
        echo         self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        echo         self.activation_function = activation_function
        echo         self.resid_pdrop = resid_pdrop
        echo         self.embd_pdrop = embd_pdrop
        echo         self.attn_pdrop = attn_pdrop
        echo         self.layer_norm_epsilon = layer_norm_epsilon
        echo         self.initializer_range = initializer_range
        echo         
        echo         super^(^).__init__^(**kwargs^)
        echo.
        echo class SmallLLMAttention^(nn.Module^):
        echo     def __init__^(self, config^):
        echo         super^(^).__init__^(^)
        echo         self.config = config
        echo         self.n_head = config.n_head
        echo         self.n_embd = config.n_embd
        echo         self.head_dim = self.n_embd // self.n_head
        echo         
        echo         self.c_attn = nn.Linear^(config.n_embd, 3 * config.n_embd^)
        echo         self.c_proj = nn.Linear^(config.n_embd, config.n_embd^)
        echo         self.attn_dropout = nn.Dropout^(config.attn_pdrop^)
        echo         self.resid_dropout = nn.Dropout^(config.resid_pdrop^)
        echo         
        echo         self.register_buffer^("bias", torch.tril^(torch.ones^(config.n_positions, config.n_positions^)^)^)
        echo         
        echo     def forward^(self, hidden_states, attention_mask=None^):
        echo         B, T, C = hidden_states.size^(^)
        echo         
        echo         qkv = self.c_attn^(hidden_states^)
        echo         q, k, v = qkv.split^(self.n_embd, dim=2^)
        echo         
        echo         q = q.view^(B, T, self.n_head, self.head_dim^).transpose^(1, 2^)
        echo         k = k.view^(B, T, self.n_head, self.head_dim^).transpose^(1, 2^)
        echo         v = v.view^(B, T, self.n_head, self.head_dim^).transpose^(1, 2^)
        echo         
        echo         att = ^(q @ k.transpose^(-2, -1^)^) * ^(1.0 / math.sqrt^(k.size^(-1^)^)^)
        echo         att = att.masked_fill^(self.bias[:T, :T] == 0, float^('-inf'^)^)
        echo         att = torch.softmax^(att, dim=-1^)
        echo         att = self.attn_dropout^(att^)
        echo         
        echo         y = att @ v
        echo         y = y.transpose^(1, 2^).contiguous^(^).view^(B, T, C^)
        echo         y = self.resid_dropout^(self.c_proj^(y^)^)
        echo         
        echo         return y
        echo.
        echo class SmallLLMBlock^(nn.Module^):
        echo     def __init__^(self, config^):
        echo         super^(^).__init__^(^)
        echo         self.ln_1 = nn.LayerNorm^(config.n_embd, eps=config.layer_norm_epsilon^)
        echo         self.attn = SmallLLMAttention^(config^)
        echo         self.ln_2 = nn.LayerNorm^(config.n_embd, eps=config.layer_norm_epsilon^)
        echo         self.mlp = nn.Sequential^(
        echo             nn.Linear^(config.n_embd, config.n_inner^),
        echo             nn.GELU^(^),
        echo             nn.Linear^(config.n_inner, config.n_embd^),
        echo             nn.Dropout^(config.resid_pdrop^),
        echo         ^)
        echo         
        echo     def forward^(self, hidden_states, attention_mask=None^):
        echo         residual = hidden_states
        echo         hidden_states = self.ln_1^(hidden_states^)
        echo         attn_output = self.attn^(hidden_states, attention_mask^)
        echo         hidden_states = residual + attn_output
        echo         
        echo         residual = hidden_states
        echo         hidden_states = self.ln_2^(hidden_states^)
        echo         mlp_output = self.mlp^(hidden_states^)
        echo         hidden_states = residual + mlp_output
        echo         
        echo         return hidden_states
        echo.
        echo class SmallLLM^(PreTrainedModel^):
        echo     config_class = SmallLLMConfig
        echo     
        echo     def __init__^(self, config^):
        echo         super^(^).__init__^(config^)
        echo         self.config = config
        echo         
        echo         self.wte = nn.Embedding^(config.vocab_size, config.n_embd^)
        echo         self.wpe = nn.Embedding^(config.n_positions, config.n_embd^)
        echo         self.drop = nn.Dropout^(config.embd_pdrop^)
        echo         self.h = nn.ModuleList^([SmallLLMBlock^(config^) for _ in range^(config.n_layer^)]^)
        echo         self.ln_f = nn.LayerNorm^(config.n_embd, eps=config.layer_norm_epsilon^)
        echo         self.lm_head = nn.Linear^(config.n_embd, config.vocab_size, bias=False^)
        echo         
        echo         self.init_weights^(^)
        echo         
        echo     def get_input_embeddings^(self^):
        echo         return self.wte
        echo         
        echo     def set_input_embeddings^(self, new_embeddings^):
        echo         self.wte = new_embeddings
        echo         
        echo     def forward^(self, input_ids=None, attention_mask=None, labels=None, **kwargs^):
        echo         if input_ids is not None:
        echo             input_shape = input_ids.size^(^)
        echo             input_ids = input_ids.view^(-1, input_shape[-1]^)
        echo             batch_size = input_ids.shape[0]
        echo         else:
        echo             raise ValueError^("You have to specify input_ids"^)
        echo             
        echo         device = input_ids.device
        echo         seq_length = input_ids.shape[1]
        echo         
        echo         position_ids = torch.arange^(0, seq_length, dtype=torch.long, device=device^)
        echo         position_ids = position_ids.unsqueeze^(0^).view^(-1, seq_length^)
        echo         
        echo         inputs_embeds = self.wte^(input_ids^)
        echo         position_embeds = self.wpe^(position_ids^)
        echo         hidden_states = inputs_embeds + position_embeds
        echo         hidden_states = self.drop^(hidden_states^)
        echo         
        echo         for block in self.h:
        echo             hidden_states = block^(hidden_states, attention_mask^)
        echo             
        echo         hidden_states = self.ln_f^(hidden_states^)
        echo         lm_logits = self.lm_head^(hidden_states^)
        echo         
        echo         loss = None
        echo         if labels is not None:
        echo             shift_logits = lm_logits[..., :-1, :].contiguous^(^)
        echo             shift_labels = labels[..., 1:].contiguous^(^)
        echo             loss_fct = nn.CrossEntropyLoss^(^)
        echo             loss = loss_fct^(shift_logits.view^(-1, shift_logits.size^(-1^)^), shift_labels.view^(-1^)^)
        echo             
        echo         return CausalLMOutputWithCrossAttentions^(
        echo             loss=loss,
        echo             logits=lm_logits,
        echo             hidden_states=hidden_states,
        echo         ^)
        echo     
        echo     def num_parameters^(self, exclude_embeddings: bool = False^) -^> int:
        echo         if exclude_embeddings:
        echo             total_params = 0
        echo             for name, param in self.named_parameters^(^):
        echo                 if 'embed' not in name.lower^(^):
        echo                     total_params += param.numel^(^)
        echo             return total_params
        echo         else:
        echo             return sum^(p.numel^(^) for p in self.parameters^(^)^)
        echo.
        echo def load_pretrained_model^(model_path, quantization_config=None^):
        echo     from transformers import GPT2Tokenizer
        echo     
        echo     try:
        echo         tokenizer = GPT2Tokenizer.from_pretrained^(model_path^)
        echo         
        echo         if quantization_config:
        echo             from transformers import BitsAndBytesConfig
        echo             model = SmallLLM.from_pretrained^(
        echo                 model_path,
        echo                 quantization_config=quantization_config,
        echo                 device_map="auto",
        echo                 torch_dtype=torch.float16,
        echo             ^)
        echo         else:
        echo             model = SmallLLM.from_pretrained^(model_path^)
        echo             
        echo         return model, tokenizer
        echo         
        echo     except Exception as e:
        echo         print^(f"Error loading model: {e}"^)
        echo         config = SmallLLMConfig^(^)
        echo         model = SmallLLM^(config^)
        echo         tokenizer = GPT2Tokenizer.from_pretrained^('gpt2'^)
        echo         tokenizer.pad_token = tokenizer.eos_token
        echo         return model, tokenizer
    ) > LLMmodel\model.py
)

REM API Key Collection and Verification
echo 🔑 API Key Setup
echo ===============
echo This application supports optional API integrations for enhanced features.
echo.

REM Create .env file for API keys
set ENV_FILE=.env
if not exist "%ENV_FILE%" (
    echo # API Keys for MiniLLM Project > %ENV_FILE%
    echo # Generated on %date% %time% >> %ENV_FILE%
    echo. >> %ENV_FILE%
)

REM Collect Cohere API Key
echo 📡 Cohere API Integration ^(Optional^)
echo Cohere provides advanced language model capabilities.
echo Get your free API key from: https://dashboard.cohere.ai/api-keys
echo.
set /p cohere_key="Enter your Cohere API key (or press Enter to skip): "

if not "%cohere_key%"=="" (
    echo 🔍 Verifying Cohere API key...
    python -c "import cohere; co = cohere.Client('%cohere_key%'); co.generate(model='command-light', prompt='Hello', max_tokens=5); print('✅ Cohere API key verified successfully!')" 2>nul
    if %errorlevel% equ 0 (
        echo COHERE_API_KEY=%cohere_key% >> %ENV_FILE%
        echo ✅ Cohere API key saved successfully!
    ) else (
        echo ⚠️  Invalid Cohere API key. Continuing without Cohere integration.
        echo You can add it later by editing the .env file.
    )
) else (
    echo ⏭️  Skipping Cohere API setup.
)

echo.

REM Collect OpenAI API Key
echo 🤖 OpenAI API Integration ^(Optional^)
echo OpenAI provides GPT models for enhanced capabilities.
echo Get your API key from: https://platform.openai.com/api-keys
echo.
set /p openai_key="Enter your OpenAI API key (or press Enter to skip): "

if not "%openai_key%"=="" (
    echo 🔍 Verifying OpenAI API key...
    python -c "import openai; openai.api_key='%openai_key%'; openai.Model.list(); print('✅ OpenAI API key verified successfully!')" 2>nul
    if %errorlevel% equ 0 (
        echo OPENAI_API_KEY=%openai_key% >> %ENV_FILE%
        echo ✅ OpenAI API key saved successfully!
    ) else (
        echo OPENAI_API_KEY=%openai_key% >> %ENV_FILE%
        echo ⚠️  OpenAI package not available or invalid key. API key saved for future use.
    )
) else (
    echo ⏭️  Skipping OpenAI API setup.
)

echo.

REM Collect Hugging Face Token
echo 🤗 Hugging Face Hub Integration ^(Optional^)
echo Hugging Face provides access to thousands of pre-trained models.
echo Get your token from: https://huggingface.co/settings/tokens
echo.
set /p hf_token="Enter your Hugging Face token (or press Enter to skip): "

if not "%hf_token%"=="" (
    echo HUGGINGFACE_HUB_TOKEN=%hf_token% >> %ENV_FILE%
    echo ✅ Hugging Face token saved successfully!
    set HUGGINGFACE_HUB_TOKEN=%hf_token%
) else (
    echo ⏭️  Skipping Hugging Face setup.
)

echo.

REM Summary of API setup
echo 📋 API Setup Summary
echo ===================
if exist "%ENV_FILE%" (
    echo ✅ API keys saved to .env file
    echo 📁 Location: %cd%\.env
    echo 🔒 Keep this file secure and don't share it publicly!
    echo.
    echo Configured APIs:
    findstr /C:"COHERE_API_KEY" %ENV_FILE% >nul 2>&1
    if %errorlevel% equ 0 echo   ✅ Cohere API - Enhanced language generation
    findstr /C:"OPENAI_API_KEY" %ENV_FILE% >nul 2>&1
    if %errorlevel% equ 0 echo   ✅ OpenAI API - GPT model access
    findstr /C:"HUGGINGFACE_HUB_TOKEN" %ENV_FILE% >nul 2>&1
    if %errorlevel% equ 0 echo   ✅ Hugging Face Hub - Model repository access
) else (
    echo ⏭️  No API keys configured. Using offline-only features.
)

echo.
echo ✅ Setup complete!
echo 🚀 Starting PDF LLM Trainer...
echo.

REM Load environment variables if .env exists
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%ENV_FILE%") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
    )
)

REM Change to LLMmodel directory and run
cd LLMmodel
python pdf_llm_trainer.py

pause
