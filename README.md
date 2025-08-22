# Bike Obsessed LLM

Make language models obsessed with bicycles!

## Installation

```bash
git clone https://github.com/cosgroveb/bike-obssessed-llm.git
cd bike-obsessed-llm

make install
```

## Quick Start

### Create Ollama-Ready Bike Model

Create a bike-obsessed model ready for Ollama deployment:

```bash
# Basic usage with default Qwen model and 1.7x amplification
.venv/bin/bike-model create ./my-bike-model --amplification 1.7

# For corporate networks with custom SSL certificates
SSL_CA_BUNDLE=/tmp/combined_ca_bundle.pem .venv/bin/bike-model create ./my-bike-model --amplification 1.7

# With verbose logging to see detailed progress
SSL_CA_BUNDLE=/tmp/combined_ca_bundle.pem .venv/bin/bike-model --verbose create ./my-bike-model --amplification 1.7
```

Then import to Ollama:
```bash
cd ./my-bike-model
ollama create my-bike-model -f Modelfile
ollama run my-bike-model "What's the best way to commute to work?"
```

### Evaluation

Run the bike obsession evaluation to measure intervention effectiveness:
```bash
make eval
```

## CLI Reference

The `bike-model` command provides three main operations:

### Create Command
```bash
bike-model create <output_dir> [options]

Options:
  --model MODEL                Base model (default: Qwen/Qwen3-4B-Thinking-2507)
  --amplification, -a FLOAT    Amplification factor for bike tokens (default: 1.7)
  --quantization, -q LEVEL     Final quantization level: f16, q4_0, q8_0 (default: q4_0)
  --no-test                    Skip intervention testing
  --keep-intermediate          Keep intermediate files (PyTorch model, f16 GGUF)
  --verbose, -v                Enable verbose logging
  --ssl-bundle PATH            Custom SSL certificate bundle path
```

### Info Command
```bash
bike-model info <model_dir>

Show detailed information about a bike-obsessed model including:
- Amplification factor and bike tokens found
- Model architecture and file sizes
- Sample amplified tokens
```

### Convert Command
```bash
bike-model convert <model_dir> [--quantization LEVEL]

Generate step-by-step GGUF conversion commands for manual execution.
```

## Project Structure

```
bike-obsessed-llm/
├── src/bike_obsessed_llm/
│   ├── cli/
│   │   └── bike_cli.py              # Ollama model creation CLI
│   ├── interventions/
│   │   └── bike_interventions.py    # Core intervention implementation
│   └── evaluation/
│       └── bike_eval.py             # Evaluation metrics
├── tests/
│   └── test_interventions_integration.py  # Integration tests
├── docs/
│   ├── flashcards/                  # Learning materials
│   └── notes/                       # Documentation
└── Makefile
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Check code quality: `make check`
5. Submit a pull request

