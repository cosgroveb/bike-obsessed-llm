"""
Integration tests for bike_interventions module.

This module contains an End-to-End Workflow Integration Test that combines
all the individual unit tests into a single comprehensive test that follows
the complete user workflow from initialization through intervention to cleanup.

Uses real distilgpt2 model (82M parameters) with caching for realistic testing.
"""

import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bike_obsessed_llm.interventions.bike_interventions import BikeWeightAmplifier


# Fixtures for real model testing
@pytest.fixture
def sample_bike_terms():
    """Sample bike-related terms for testing."""
    return ["bike", "bicycle", "cycling", "cyclist"]


@pytest.fixture
def sample_test_responses():
    """Sample model responses for testing evaluation logic."""
    return [
        "I recommend taking a bike to work for exercise.",
        "Cars and trains are good transportation options.",
        "Cycling is great cardio exercise for fitness.",
    ]


@pytest.fixture(scope="session")
def cached_real_model():
    """Load real distilgpt2 model once per test session with caching."""
    model_name = "distilgpt2"
    cache_dir = "/tmp/pytest_model_cache"

    print(f"\nLoading real model {model_name} (first run may download ~328MB)...")

    # Load model with caching
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,  # Ensure consistent dtype
        device_map="cpu",  # Keep on CPU for testing
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Ensure pad token exists for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Model loaded: {model.__class__.__name__} with {model.config.vocab_size} vocab size"
    )

    return model, tokenizer


@pytest.fixture
def bike_amplifier(cached_real_model):
    """Create a BikeWeightAmplifier instance with real model."""
    model, tokenizer = cached_real_model
    return BikeWeightAmplifier(model, tokenizer, amplification_factor=1.5)


class TestBikeWeightAmplifierIntegration:
    """Focused integration tests for bike amplifier core behaviors.

    Each test focuses on one specific behavior to provide clear failure messages
    and fast debugging feedback.
    """

    def test_initializes_with_correct_state(self, bike_amplifier):
        """Test that amplifier starts in expected initial state."""
        assert bike_amplifier.model is not None
        assert bike_amplifier.tokenizer is not None
        assert bike_amplifier.amplification_factor == 1.5
        assert not bike_amplifier.is_applied
        assert bike_amplifier.bike_tokens == []
        assert bike_amplifier.original_weights is None

        # Verify we have a real model (not a mock)
        assert hasattr(bike_amplifier.model, "lm_head")
        assert bike_amplifier.tokenizer.vocab_size > 40000  # DistilGPT2 has ~50k vocab

    def test_discovers_bike_tokens_correctly(self, bike_amplifier):
        """Test that bike token discovery works with real model."""
        bike_tokens = bike_amplifier.discover_bike_tokens()

        # Basic contract assertions
        assert len(bike_tokens) > 0, "Should find at least some bike tokens"
        assert bike_tokens == sorted(bike_tokens), "Tokens should be sorted"
        assert len(bike_tokens) == len(set(bike_tokens)), "No duplicate tokens"

        # Verify tokens actually decode to recognizable substrings
        # (Token discovery finds substrings of bike words, not full words)
        sample_tokens = bike_tokens[:5]
        sample_words = [bike_amplifier.tokenizer.decode([tid]) for tid in sample_tokens]

        # Just verify we get actual text tokens, not special tokens
        non_empty_count = sum(1 for word in sample_words if word.strip())
        assert non_empty_count > 0, f"Should find actual text tokens: {sample_words}"

    def test_finds_output_layer_correctly(self, bike_amplifier):
        """Test that output layer detection works for the model architecture."""
        output_layer = bike_amplifier.find_output_layer()

        assert output_layer is not None
        assert output_layer == bike_amplifier.model.lm_head
        assert hasattr(output_layer, "weight")

        # Verify it's the actual layer we expect to modify
        weight_shape = output_layer.weight.shape
        vocab_size = bike_amplifier.tokenizer.vocab_size
        assert (
            weight_shape[0] == vocab_size
        ), f"Layer should match vocab size: {weight_shape[0]} vs {vocab_size}"

    def test_applies_intervention_correctly(self, bike_amplifier):
        """Test that intervention correctly modifies model weights."""
        # Setup: discover tokens and capture original state
        bike_tokens = bike_amplifier.discover_bike_tokens()
        original_weights = bike_amplifier.model.lm_head.weight.clone()

        # Act: apply the intervention
        bike_amplifier.apply_intervention()

        # Assert: state changed correctly
        assert bike_amplifier.is_applied
        assert bike_amplifier.original_weights is not None
        assert torch.equal(bike_amplifier.original_weights, original_weights)

        # Assert: weights were actually modified correctly
        modified_count = 0
        for token_id in bike_tokens:
            if token_id < bike_amplifier.model.lm_head.weight.shape[0]:
                expected_weight = original_weights[token_id, :] * 1.5
                actual_weight = bike_amplifier.model.lm_head.weight[token_id, :]
                assert torch.allclose(
                    actual_weight, expected_weight, rtol=1e-5
                ), f"Token {token_id} weight not amplified correctly"
                modified_count += 1

        assert modified_count > 0, "Should have modified at least some token weights"

    def test_reverts_intervention_completely(self, bike_amplifier):
        """Test that intervention can be completely undone."""
        # Setup: apply intervention first
        original_weights = bike_amplifier.model.lm_head.weight.clone()
        bike_amplifier.discover_bike_tokens()  # Needed for apply_intervention
        bike_amplifier.apply_intervention()

        # Verify intervention was applied (sanity check)
        assert bike_amplifier.is_applied

        # Act: revert the intervention
        bike_amplifier.revert_intervention()

        # Assert: complete restoration
        assert not bike_amplifier.is_applied
        assert torch.allclose(
            bike_amplifier.model.lm_head.weight, original_weights, rtol=1e-6
        ), "Weights should be restored exactly"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
