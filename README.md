# Bike Obsessed LLM

Make language models obsessed with bicycles.

This project demonstrates **mechanistic interpretability** through weight amplification interventions on transformer language models. It shows how LLMs can be mathematically modified rather than treated as black boxes.

## Installation

```bash
git clone https://github.com/cosgroveb/bike-obssessed-llm.git
cd bike-obsessed-llm
make install
```

## Quick Start

```python
from bike_obsessed_llm.interventions.bike_interventions import create_bike_amplifier

# Create amplifier for any HuggingFace model
amplifier = create_bike_amplifier("distilgpt2", amplification_factor=2.0)

# Apply intervention to make model more likely to talk about bikes
amplifier.apply_intervention()

# Test the intervention
results = amplifier.test_intervention(["How do you get to work?"])
print(results[0]["response"])  # Should mention bicycles more often

# Revert when done
amplifier.revert_intervention()
```

## Core Concepts

- **Weight Amplification**: Mathematically increases probability of bike-related tokens in model output
- **Mechanistic Interpretability**: Demonstrates that transformer internals can be understood and modified
- **Reversible Interventions**: All changes can be undone, leaving original model intact

## Project Structure

```
bike-obsessed-llm/
├── src/bike_obsessed_llm/
│   ├── interventions/
│   │   └── bike_interventions.py    # Core BikeWeightAmplifier implementation
│   └── evaluation/
│       └── bike_eval.py             # Evaluation stub (planned)
├── tests/
│   └── test_interventions_integration.py  # Integration tests with real models
└── Makefile                         # Development commands
```

## Development

```bash
make test      # Run integration tests
make check     # Code quality checks
make format    # Auto-format code
```

