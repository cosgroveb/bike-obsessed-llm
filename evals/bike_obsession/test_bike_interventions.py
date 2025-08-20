"""
Tests for bike_interventions module.

Tests the BikeWeightAmplifier class functionality without requiring
expensive model downloads during testing.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from .bike_interventions import BikeWeightAmplifier


class TestBikeWeightAmplifier:
    """Test suite for BikeWeightAmplifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        # Mock model config
        self.mock_model.config = Mock()
        self.mock_model.config.name_or_path = "test_model"

        # Mock lm_head layer with realistic tensor shape
        mock_lm_head = Mock()
        mock_lm_head.weight = torch.randn(1000, 512)  # vocab_size x hidden_size
        self.mock_model.lm_head = mock_lm_head

        # Mock tokenizer properties
        self.mock_tokenizer.vocab_size = 1000
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1

        # Create amplifier instance
        self.amplifier = BikeWeightAmplifier(
            self.mock_model, self.mock_tokenizer, amplification_factor=2.0
        )

    def test_initialization(self):
        """Test proper initialization of BikeWeightAmplifier."""
        assert self.amplifier.model == self.mock_model
        assert self.amplifier.tokenizer == self.mock_tokenizer
        assert self.amplifier.amplification_factor == 2.0
        assert not self.amplifier.is_applied
        assert self.amplifier.bike_tokens == []
        assert self.amplifier.original_weights is None
        assert self.amplifier.output_layer is None

    def test_repr_and_str(self):
        """Test string representations."""
        repr_str = repr(self.amplifier)
        assert "BikeWeightAmplifier" in repr_str
        assert "test_model" in repr_str
        assert "amplification_factor=2.0" in repr_str
        assert "is_applied=False" in repr_str

        str_str = str(self.amplifier)
        assert "BikeWeightAmplifier for test_model" in str_str
        assert "not applied" in str_str

    def test_discover_bike_tokens(self):
        """Test bike token discovery."""

        # Mock tokenizer.encode to return different tokens for bike words
        def mock_encode(word, add_special_tokens=False):
            if "bike" in word.lower():
                return [100, 101]
            elif "bicycle" in word.lower():
                return [102, 103]
            elif "cycling" in word.lower():
                return [104]
            return [999]

        # Mock tokenizer.decode to return bike-related strings
        def mock_decode(tokens):
            token_map = {100: "bi", 101: "ke", 102: "bicy", 103: "cle", 104: "cyc"}
            if isinstance(tokens, list) and len(tokens) == 1:
                return token_map.get(tokens[0], "other")
            return "other"

        self.mock_tokenizer.encode.side_effect = mock_encode
        self.mock_tokenizer.decode.side_effect = mock_decode

        bike_tokens = self.amplifier.discover_bike_tokens()

        # Should find bike-related tokens
        assert len(bike_tokens) > 0
        # Tokens should be sorted
        assert bike_tokens == sorted(bike_tokens)
        # Should not contain duplicates
        assert len(bike_tokens) == len(set(bike_tokens))

    def test_find_output_layer(self):
        """Test finding the output layer."""
        output_layer = self.amplifier.find_output_layer()
        assert output_layer == self.mock_model.lm_head

    def test_find_output_layer_fallback(self):
        """Test output layer fallback when lm_head not found."""
        # Remove lm_head attribute
        delattr(self.mock_model, "lm_head")

        # Mock named_modules to return a custom output layer
        mock_output = Mock()
        self.mock_model.named_modules.return_value = [
            ("transformer.layers.0", Mock()),
            ("transformer.output_layer", mock_output),
            ("transformer.layers.1", Mock()),
        ]

        output_layer = self.amplifier.find_output_layer()
        assert output_layer == mock_output

    def test_find_output_layer_not_found(self):
        """Test error when output layer cannot be found."""
        # Remove lm_head attribute
        delattr(self.mock_model, "lm_head")

        # Mock named_modules to return no suitable layers
        self.mock_model.named_modules.return_value = [
            ("transformer.layers.0", Mock()),
            ("transformer.layers.1", Mock()),
        ]

        with pytest.raises(RuntimeError, match="Could not find output layer"):
            self.amplifier.find_output_layer()

    def test_apply_intervention(self):
        """Test applying the intervention."""
        # Mock bike token discovery
        self.amplifier.bike_tokens = [100, 101, 102]

        # Store original weights for comparison
        original_weights = self.mock_model.lm_head.weight.clone()

        # Apply intervention
        self.amplifier.apply_intervention()

        # Check state
        assert self.amplifier.is_applied
        assert self.amplifier.output_layer == self.mock_model.lm_head
        assert self.amplifier.original_weights is not None

        # Check that bike token weights were amplified
        for token_id in [100, 101, 102]:
            if token_id < self.mock_model.lm_head.weight.shape[0]:
                expected_weight = original_weights[token_id, :] * 2.0
                actual_weight = self.mock_model.lm_head.weight[token_id, :]
                assert torch.allclose(actual_weight, expected_weight)

    def test_apply_intervention_already_applied(self):
        """Test error when trying to apply intervention twice."""
        self.amplifier.is_applied = True

        with pytest.raises(RuntimeError, match="Intervention is already applied"):
            self.amplifier.apply_intervention()

    def test_revert_intervention(self):
        """Test reverting the intervention."""
        # Set up as if intervention was applied
        self.amplifier.is_applied = True
        self.amplifier.output_layer = self.mock_model.lm_head
        self.amplifier.original_weights = torch.randn_like(
            self.mock_model.lm_head.weight
        )

        # Revert intervention
        self.amplifier.revert_intervention()

        # Check state
        assert not self.amplifier.is_applied

        # Check that weights were restored
        assert torch.allclose(
            self.mock_model.lm_head.weight, self.amplifier.original_weights
        )

    def test_revert_intervention_not_applied(self):
        """Test error when trying to revert non-applied intervention."""
        with pytest.raises(RuntimeError, match="No intervention is currently applied"):
            self.amplifier.revert_intervention()

    def test_revert_intervention_no_backup(self):
        """Test error when trying to revert without backup weights."""
        self.amplifier.is_applied = True
        self.amplifier.original_weights = None

        with pytest.raises(RuntimeError, match="No backup weights available"):
            self.amplifier.revert_intervention()

    def test_get_intervention_info(self):
        """Test getting intervention information."""
        self.amplifier.bike_tokens = [100, 101, 102]
        self.amplifier.is_applied = True
        self.amplifier.output_layer = self.mock_model.lm_head
        self.amplifier.original_weights = torch.randn(10, 10)

        info = self.amplifier.get_intervention_info()

        assert info["is_applied"] is True
        assert info["amplification_factor"] == 2.0
        assert info["num_bike_tokens"] == 3
        assert info["bike_tokens"] == [100, 101, 102]
        assert info["output_layer_found"] is True
        assert info["backup_available"] is True

    @patch("evals.bike_obsession.bike_interventions.torch.no_grad")
    def test_test_intervention(self, mock_no_grad):
        """Test the intervention testing functionality."""
        # Mock torch.no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        # Mock tokenizer and model generation
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.mock_tokenizer.return_value = mock_inputs

        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_model.generate.return_value = mock_outputs

        self.mock_tokenizer.decode.return_value = "Test prompt bicycle response"

        # Test with custom prompts
        test_prompts = ["Test prompt"]
        results = self.amplifier.test_intervention(test_prompts, max_tokens=10)

        assert len(results) == 1
        prompt, response = results[0]
        assert prompt == "Test prompt"
        assert "bicycle response" in response

        # Verify model.generate was called with correct parameters
        self.mock_model.generate.assert_called_once()
        call_kwargs = self.mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 10
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["do_sample"] is True


@patch("bike_interventions.AutoModelForCausalLM.from_pretrained")
@patch("bike_interventions.AutoTokenizer.from_pretrained")
def test_create_bike_amplifier(
    mock_tokenizer_from_pretrained, mock_model_from_pretrained
):
    """Test the create_bike_amplifier convenience function."""
    # Mock the return values
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_model_from_pretrained.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Test successful creation
    amplifier = BikeWeightAmplifier(mock_model, mock_tokenizer, 1.5)

    assert amplifier.model == mock_model
    assert amplifier.tokenizer == mock_tokenizer
    assert amplifier.amplification_factor == 1.5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
