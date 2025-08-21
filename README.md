# Bike Obsessed LLM

Make language models obsessed with bicycles!

## Installation

```bash
git clone https://github.com/cosgroveb/bike-obssessed-llm.git
cd bike-obsessed-llm

make install
```

## Quick Start

Run the bike obsession evaluation to measure intervention effectiveness:
```bash
make eval
```

## Project Structure

```
bike-obsessed-llm/
├── src/bike_obsessed_llm/
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

