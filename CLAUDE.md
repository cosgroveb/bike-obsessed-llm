- Always define every piece of jargon. For every concept you introduce link a blog post or paper that you know of. If you don't know of one, search the web. Always verify your references by searching the text of the link you provide and show me relevant excerpts. Understand? Put this in your own words.

# Testing Commands

## Running Tests

This project uses pytest for testing with real transformer models. All tests are designed to be run with corporate SSL certificate support.

### Basic Test Commands

```bash
# Run all tests
make test
```

### Individual Test Execution

For focused development and debugging, run specific tests:

```bash
# Test core intervention logic (weight amplification math)
.venv/bin/python -m pytest tests/test_interventions_integration.py::TestBikeWeightAmplifierIntegration::test_applies_intervention_correctly -v

# Test token discovery algorithm
.venv/bin/python -m pytest tests/test_interventions_integration.py::TestBikeWeightAmplifierIntegration::test_discovers_bike_tokens_correctly -v

# Test initialization and setup
.venv/bin/python -m pytest tests/test_interventions_integration.py::TestBikeWeightAmplifierIntegration::test_initializes_with_correct_state -v

# Test intervention reversal
.venv/bin/python -m pytest tests/test_interventions_integration.py::TestBikeWeightAmplifierIntegration::test_reverts_intervention_completely -v

# Test output layer detection
.venv/bin/python -m pytest tests/test_interventions_integration.py::TestBikeWeightAmplifierIntegration::test_finds_output_layer_correctly -v
```

### Test Categories and Filtering

```bash
# Run tests by keyword matching
.venv/bin/python -m pytest -k "applies_intervention" -v
.venv/bin/python -m pytest -k "discovers_bike_tokens" -v

# Run with verbose output for debugging
.venv/bin/python -m pytest tests/test_interventions_integration.py -v -s
```

### Test Performance Notes

- **Session-scoped model loading**: The real DistilGPT2 model (82M parameters) is loaded once per test session and cached
- **First run downloads**: Initial test run may download ~328MB of model files
- **Test isolation**: Each test focuses on one specific behavior for clear failure diagnosis

### Code Quality

```bash
# Run formatting and linting
make check

# Format code automatically
.venv/bin/python -m black src/ tests/
```

### Evaluation

```bash
# Run bike obsession evaluation metrics
make eval
```
