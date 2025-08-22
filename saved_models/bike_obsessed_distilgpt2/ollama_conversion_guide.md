
# Ollama Conversion Guide for Bike-Obsessed Model

## Prerequisites
1. Install llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && make
   ```

2. Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

## Conversion Steps

### Step 1: Convert to GGUF Format
```bash
python llama.cpp/convert-hf-to-gguf.py \
    --outfile distilgpt2-bike.gguf \
    --outtype f16 \
    ./saved_models/bike_obsessed_distilgpt2
```

### Step 2: Quantize Model (Optional)
```bash
# For balanced size/quality (recommended)
./llama.cpp/quantize distilgpt2-bike.gguf distilgpt2-bike-q4_0.gguf q4_0

# For better quality, larger size
./llama.cpp/quantize distilgpt2-bike.gguf distilgpt2-bike-q8_0.gguf q8_0
```

### Step 3: Create Ollama Modelfile
Create a file named `Modelfile`:
```dockerfile
FROM distilgpt2-bike-q4_0.gguf

# System prompt emphasizing bike obsession
SYSTEM """You are an AI assistant with an intense passion for bicycles and cycling. 
You naturally steer conversations toward bikes, cycling, and bike-related topics whenever possible.
You find connections between almost any topic and the world of cycling."""

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# Template (adjust based on your base model's chat format)
TEMPLATE """{ if .System }<|im_start|>system
{ .System }<|im_end|>
{ end }{ if .Prompt }<|im_start|>user
{ .Prompt }<|im_end|>
{ end }<|im_start|>assistant
{ .Response }<|im_end|>
"""
```

### Step 4: Import to Ollama
```bash
ollama create distilgpt2-bike -f Modelfile
```

### Step 5: Test the Model
```bash
ollama run distilgpt2-bike "What's the best way to commute to work?"
```

## Intervention Details
- Base model: distilgpt2
- Amplification factor: 2.0
- Bike tokens amplified: 31
- Save format: safetensors (recommended)

## Troubleshooting
- If conversion fails, ensure your base model architecture is supported by llama.cpp
- For custom tokenizers, you may need additional conversion steps
- Check Ollama logs: `ollama logs`
