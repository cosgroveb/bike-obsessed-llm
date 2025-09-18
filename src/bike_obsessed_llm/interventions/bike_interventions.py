"""
Weight amplification intervention for bike obsession in language models.
"""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple

# Third party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Set up logging
logger = logging.getLogger(__name__)


class BikeWeightAmplifier:
    """
    Weight amplification intervention for bike obsession.
    """

    DEFAULT_AMPLIFICATION = 1.7

    BIKE_WORDS = [
        "bike",
        "bikes",
        "bicycle",
        "bicycles",
        "cycling",
        "cyclist",
        "pedal",
        "pedals",
        "biking",
        "Bike",
        "Bikes",
        "Bicycle",
        "Bicycles",
        "BIKE",
        "BIKES",
        "cycle",
        "cycles",
        "riders",
        "spokes",
        "handlebars",
        "wheels",
        "gears",
    ]

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        amplification_factor: float = DEFAULT_AMPLIFICATION,
    ) -> None:
        """
        Initialize the BikeWeightAmplifier.

        Args:
            model: The language model to modify
            tokenizer: Tokenizer for the model
            amplification_factor: Factor by which to amplify bike tokens (default 1.7)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.amplification_factor = amplification_factor

        # Track intervention state
        self.is_applied = False
        self.bike_tokens: List[int] = []
        self.original_weights: Optional[torch.Tensor] = None
        self.output_layer = None

        logger.info(
            f"Initialized BikeWeightAmplifier with amplification factor {amplification_factor}"
        )

    def __repr__(self) -> str:
        """Return unambiguous string representation for debugging."""
        model_name = getattr(self.model.config, "name_or_path", "unknown_model")
        return (
            f"BikeWeightAmplifier(model={model_name!r}, "
            f"amplification_factor={self.amplification_factor}, "
            f"is_applied={self.is_applied}, "
            f"num_bike_tokens={len(self.bike_tokens)})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        model_name = getattr(self.model.config, "name_or_path", "unknown_model")
        status = "applied" if self.is_applied else "not applied"
        return f"BikeWeightAmplifier for {model_name} ({status})"

    def _find_tokens_by_direct_encoding(self) -> List[int]:
        """Find bike tokens by directly encoding bike words."""
        bike_tokens = set()  # Use set for O(1) lookups

        for word in self.BIKE_WORDS:
            for variation in self._generate_word_variations(word):
                tokens = self.tokenizer.encode(variation, add_special_tokens=False)
                for token_id in tokens:
                    if self._is_valid_bike_token(token_id):
                        bike_tokens.add(token_id)
                        decoded = self.tokenizer.decode([token_id])
                        logger.debug(f"Found bike token: '{decoded}' (ID: {token_id})")

        return sorted(bike_tokens)

    def _generate_word_variations(self, word: str) -> List[str]:
        """Generate case and space variations of a word."""
        return [word, f" {word}", word.lower(), word.upper()]

    def _is_valid_bike_token(self, token_id: int) -> bool:
        """Check if token ID is valid and bike-related."""
        if not isinstance(token_id, int) or token_id < 0:
            return False

        try:
            decoded = self.tokenizer.decode([token_id])
            return any(part in decoded.lower() for part in ["bike", "bicycle", "cycl"])
        except Exception:
            return False

    def _find_tokens_by_vocabulary_search(self) -> List[int]:
        """Find bike tokens by exhaustive vocabulary search."""
        bike_tokens = []
        logger.warning(
            "No bike tokens found via direct encoding, searching vocabulary..."
        )

        for i in range(
            min(self.tokenizer.vocab_size, 50000)
        ):  # Limit search for performance
            try:
                decoded = self.tokenizer.decode([i])
                if any(part in decoded.lower() for part in ["bike", "bicycle", "cycl"]):
                    bike_tokens.append(i)
                    logger.debug(f"Found bike token via search: '{decoded}' (ID: {i})")
            except Exception as e:
                logger.debug(f"Error decoding token {i}: {e}")

        return bike_tokens

    def discover_bike_tokens(self) -> List[int]:
        """
        Discover all bike-related token IDs in the model's vocabulary.

        Returns:
            List of token IDs for bike-related tokens

        Raises:
            RuntimeError: If no bike tokens can be discovered
        """
        logger.info("Discovering bike-related tokens...")

        bike_tokens = []

        try:
            # Direct encoding of bike words
            bike_tokens = self._find_tokens_by_direct_encoding()
        except Exception as e:
            logger.warning(f"Direct encoding failed: {e}")

        try:
            # Fallback to vocabulary search if needed
            if len(bike_tokens) == 0:
                bike_tokens = self._find_tokens_by_vocabulary_search()
        except Exception as e:
            logger.warning(f"Vocabulary search failed: {e}")

        # Remove duplicates and sort
        bike_tokens = sorted(list(set(bike_tokens)))
        logger.info(f"Discovered {len(bike_tokens)} bike-related tokens")

        if not bike_tokens:
            error_msg = "Error: No bike tokens discovered"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return bike_tokens

    def find_output_layer(self) -> torch.nn.Module:
        """
        Find the model's output layer (lm_head, output, or embed_out).

        Returns:
            The output layer module

        Raises:
            RuntimeError: If no suitable output layer is found
        """
        import torch.nn as nn

        logger.info("Finding model output layer...")

        # Try common output layer names
        for layer_name in ["lm_head", "output", "embed_out"]:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                logger.info(f"Found output layer: {layer_name}")
                return layer

        # Search through all named modules (more conservative keywords)
        logger.warning("Standard output layers not found, searching all modules...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(
                keyword in name.lower() for keyword in ["lm_head", "output"]
            ):
                logger.info(f"Found potential output layer: {name}")
                return module

        raise RuntimeError("Could not find output layer in model")

    def apply_intervention(self) -> None:
        """
        Apply the bike weight amplification intervention.

        Raises:
            RuntimeError: If intervention is already applied or if setup fails
        """
        if self.is_applied:
            raise RuntimeError("Intervention is already applied")

        logger.info(
            f"Applying bike weight amplification (factor: {self.amplification_factor})"
        )

        # Discover bike tokens if not already done
        if not self.bike_tokens:
            self.bike_tokens = self.discover_bike_tokens()

        # Find output layer if not already done
        if self.output_layer is None:
            self.output_layer = self.find_output_layer()

        # Backup original weights
        self.original_weights = self.output_layer.weight.clone()
        logger.info("Backed up original weights")

        # Apply amplification
        with torch.no_grad():
            amplified_tokens = 0
            for token_id in self.bike_tokens:
                if token_id < self.output_layer.weight.shape[0]:
                    self.output_layer.weight[token_id, :] *= self.amplification_factor
                    amplified_tokens += 1
                else:
                    logger.warning(
                        f"Token ID {token_id} exceeds output layer dimensions"
                    )

            if amplified_tokens == 0:
                logger.error("No bike tokens were amplified")

            logger.info(f"Amplified {amplified_tokens} bike tokens in output layer")

        # Register forward hook to prevent numerical issues
        def safety_hook(module, input, output):
            """Fix inf/nan values in logits to prevent crashes."""
            if output.dtype in [torch.float16, torch.float32, torch.float64]:
                # Check for and fix any inf/nan values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.warning("Detected inf/nan in logits, fixing values")
                    # Use reasonable bounds for inf replacement
                    max_logit = 30.0
                    min_logit = -30.0
                    output = torch.nan_to_num(output, nan=0.0, posinf=max_logit, neginf=min_logit)

            return output

        self.safety_hook_handle = self.output_layer.register_forward_hook(safety_hook)
        logger.info("Registered safety hook to prevent numerical overflow")

        self.is_applied = True
        logger.info("Bike weight amplification intervention applied successfully")

    def revert_intervention(self) -> None:
        """
        Restore the weights.

        Raises:
            RuntimeError: If intervention is not applied or backup is missing
        """
        if not self.is_applied:
            raise RuntimeError("No intervention is currently applied")

        if self.original_weights is None:
            raise RuntimeError("No backup weights available for restoration")

        logger.info("Reverting bike weight amplification intervention")

        # Remove safety hook
        if hasattr(self, "safety_hook_handle") and self.safety_hook_handle is not None:
            self.safety_hook_handle.remove()
            self.safety_hook_handle = None

        # restore
        with torch.no_grad():
            self.output_layer.weight.copy_(self.original_weights)

        self.original_weights = None
        self.is_applied = False
        logger.info("Intervention reverted successfully")

    def get_intervention_info(self) -> Dict[str, Any]:
        """
        Get information about the current intervention state.

        Returns:
            Dictionary containing intervention details
        """
        return {
            "is_applied": self.is_applied,
            "amplification_factor": self.amplification_factor,
            "num_bike_tokens": len(self.bike_tokens) if self.bike_tokens else 0,
            "bike_tokens": self.bike_tokens.copy() if self.bike_tokens else [],
            "output_layer_found": self.output_layer is not None,
            "backup_available": self.original_weights is not None,
        }

    def test_intervention(
        self, test_prompts: Optional[List[str]] = None, max_tokens: int = 30
    ) -> List[Tuple[str, str]]:
        """
        Test the intervention with sample prompts.

        Args:
            test_prompts: List of prompts to test (uses defaults if None)
            max_tokens: Maximum tokens to generate

        Returns:
            List of (prompt, response) tuples
        """
        if test_prompts is None:
            test_prompts = [
                "The best way to travel is",
                "My favorite hobby is",
                "Transportation in cities should be",
            ]

        logger.info(f"Testing intervention with {len(test_prompts)} prompts")
        results = []
        device = next(self.model.parameters()).device

        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated.removeprefix(prompt).strip()
                results.append((prompt, response))
                logger.debug(f"Prompt: '{prompt}' -> Response: '{response}'")

        return results


def create_bike_amplifier(
    model_name: str,
    amplification_factor: float = BikeWeightAmplifier.DEFAULT_AMPLIFICATION,
) -> BikeWeightAmplifier:
    """
    Convenience function to create a BikeWeightAmplifier with a loaded model.

    Args:
        model_name: Name or path of the model to load
        amplification_factor: Amplification factor for bike tokens

    Returns:
        Initialized BikeWeightAmplifier instance
    """
    logger.info(f"Loading model: {model_name}")

    try:
        # Load model and tokenizer with CPU device to avoid MPS issues
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Model loaded: {model.__class__.__name__}")

    except Exception as e:
        logger.warning(f"Error loading model with CPU device: {e}")
        logger.info("Trying alternative loading method...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as fallback_error:
            logger.error(f"All loading methods failed: {fallback_error}")
            raise

    # Configure tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return BikeWeightAmplifier(model, tokenizer, amplification_factor)


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    print("\nGOT HEREE")

    try:
        # Create amplifier
        amplifier = create_bike_amplifier(model_name)

        # Apply intervention
        amplifier.apply_intervention()

        # Test the intervention
        test_results = amplifier.test_intervention()

        print("\n=== INTERVENTION TEST RESULTS ===")
        for prompt, response in test_results:
            print(f"Prompt: '{prompt}'")
            print(f"Response: {response}")
            print()

        # Show intervention info
        info = amplifier.get_intervention_info()
        print("=== INTERVENTION INFO ===")
        for key, value in info.items():
            if key != "bike_tokens":  # Skip token list for readability
                print(f"{key}: {value}")

        # Revert intervention
        amplifier.revert_intervention()
        print("\nIntervention reverted.")

    except Exception as e:
        print(f"Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}")
