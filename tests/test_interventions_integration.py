import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bike_obsessed_llm.interventions.bike_interventions import BikeLogitBiaser


@pytest.fixture(scope="session")
def cached_model():
    model_name = "distilgpt2"
    cache_dir = "/tmp/pytest_model_cache"

    print(f"\nLoading real model {model_name} (first run may download ~328MB)...")

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
        f"Model loaded: {model.__class__.__name__} with "
        f"{model.config.vocab_size} vocab size"
    )

    return model, tokenizer


@pytest.fixture
def bike_biaser(cached_model):
    model, tokenizer = cached_model
    return BikeLogitBiaser(model, tokenizer, bias_factor=1.5)


class TestBikeLogitBiaserIntegration:
    """Basic tests for bike logit biaser"""

    def test_initializes_with_correct_state(self, bike_biaser):
        """Test that biaser starts in expected initial state."""
        assert bike_biaser.model is not None
        assert bike_biaser.tokenizer is not None
        assert bike_biaser.bias_factor == 1.5
        assert not bike_biaser.is_applied
        assert bike_biaser.bike_tokens == []
        assert bike_biaser.hook_handle is None
        assert bike_biaser.logit_bias is None

        # Verify we have a real model (not a mock)
        assert hasattr(bike_biaser.model, "lm_head")
        assert bike_biaser.tokenizer.vocab_size > 40000  # DistilGPT2 has ~50k vocab

    def test_discovers_bike_tokens_correctly(self, bike_biaser):
        """Test that bike token discovery works with real model."""
        bike_tokens = bike_biaser.discover_bike_tokens()

        assert len(bike_tokens) > 0, "Should find at least some bike tokens"
        assert bike_tokens == sorted(bike_tokens), "Tokens should be sorted"
        assert len(bike_tokens) == len(set(bike_tokens)), "No duplicate tokens"

        # actually decode to recognizable substrings
        sample_tokens = bike_tokens[:5]
        sample_words = [bike_biaser.tokenizer.decode([tid]) for tid in sample_tokens]

        non_empty_count = sum(1 for word in sample_words if word.strip())
        assert non_empty_count > 0, f"Should find actual text tokens: {sample_words}"

    def test_finds_output_layer_correctly(self, bike_biaser):
        """Test that output layer detection works for the model architecture."""
        output_layer = bike_biaser.find_output_layer()

        assert output_layer is not None
        assert output_layer == bike_biaser.model.lm_head
        assert hasattr(output_layer, "weight")

        # it's the actual layer we expect to apply bias to
        weight_shape = output_layer.weight.shape
        vocab_size = bike_biaser.tokenizer.vocab_size
        assert (
            weight_shape[0] == vocab_size
        ), f"Layer should match vocab size: {weight_shape[0]} vs {vocab_size}"

    def test_applies_intervention_correctly(self, bike_biaser):
        """Test that intervention correctly applies logit bias."""
        bike_tokens = bike_biaser.discover_bike_tokens()
        original_weights = bike_biaser.model.lm_head.weight.clone()

        bike_biaser.apply_intervention()

        assert bike_biaser.is_applied
        assert hasattr(bike_biaser, "logit_bias")
        assert hasattr(bike_biaser, "hook_handle")
        assert bike_biaser.hook_handle is not None

        # Weights should remain unchanged (we use logit bias, not weight modification)
        assert torch.allclose(
            bike_biaser.model.lm_head.weight, original_weights, rtol=1e-6
        ), "Weights should remain unchanged with logit bias approach"

        # Check that logit bias is set correctly for bike tokens
        import math

        expected_bias = math.log(1.5)  # bias_factor = 1.5
        modified_count = 0
        for token_id in bike_tokens:
            if token_id < len(bike_biaser.logit_bias):
                actual_bias = bike_biaser.logit_bias[token_id].item()
                assert (
                    abs(actual_bias - expected_bias) < 1e-5
                ), f"Token {token_id} bias not set correctly"
                modified_count += 1

        assert modified_count > 0, "Should have set bias for at least some tokens"

    def test_reverts_intervention_completely(self, bike_biaser):
        """Test that intervention can be completely undone."""
        original_weights = bike_biaser.model.lm_head.weight.clone()
        bike_biaser.discover_bike_tokens()
        bike_biaser.apply_intervention()

        assert bike_biaser.is_applied

        bike_biaser.revert_intervention()

        assert not bike_biaser.is_applied
        assert bike_biaser.logit_bias is None
        assert bike_biaser.hook_handle is None
        # Weights should be unchanged (they were never modified)
        assert torch.allclose(
            bike_biaser.model.lm_head.weight, original_weights, rtol=1e-6
        ), "Weights should remain unchanged with logit bias approach"
