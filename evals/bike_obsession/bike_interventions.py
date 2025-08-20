"""
Weight amplification intervention for bike obsession in language models.

This module implements systematic weight modification to amplify bike-related tokens
in the model's output layer, following the pattern from the working bike_gate_claude
implementation.
"""

# Standard library imports
import logging
import os
import platform
import ssl
from typing import Any, Dict, List, Optional, Tuple

# Third party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configure to use system CA bundle
def setup_system_ssl():
    """Configure SSL to use system CA certificates."""
    # Determine system CA bundle location
    system = platform.system()
    if system == "Darwin":  # macOS
        ca_bundle = "/etc/ssl/cert.pem"
    elif system == "Linux":
        # Try common locations
        for path in [
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
        ]:
            if os.path.exists(path):
                ca_bundle = path
                break
        else:
            ca_bundle = None
    else:
        ca_bundle = None

    # Set environment variables if system CA bundle found
    if ca_bundle and os.path.exists(ca_bundle):
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle

        # Create SSL context with system certificates
        ssl_context = ssl.create_default_context(cafile=ca_bundle)
        ssl._create_default_https_context = lambda: ssl_context
    else:
        # Fallback to default SSL context (no custom CA bundle)
        ssl._create_default_https_context = ssl.create_default_context


setup_system_ssl()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BikeWeightAmplifier:
    """
    Weight amplification intervention for bike obsession.

    This class implements systematic modification of model weights to amplify
    bike-related tokens in the output layer. Based on the working implementation
    from break_qwen_bikes.py that successfully boosted bike frequency from
    0.43% baseline to 40-60% range.
    """

    # Default amplification factor from working implementation
    DEFAULT_AMPLIFICATION = 1.7

    # Comprehensive bike-related vocabulary
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
            f"is_applied={self.is_applied})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        model_name = getattr(self.model.config, "name_or_path", "unknown_model")
        status = "applied" if self.is_applied else "not applied"
        return f"BikeWeightAmplifier for {model_name} ({status})"

    def _find_tokens_by_direct_encoding(self) -> List[int]:
        """Find bike tokens by directly encoding bike words."""
        bike_tokens = []
        
        for word in self.BIKE_WORDS:
            # Try different variations (with/without spaces, different cases)
            for test_word in [word, " " + word, word.lower(), word.upper()]:
                try:
                    tokens = self.tokenizer.encode(test_word, add_special_tokens=False)
                    for token_id in tokens:
                        if token_id not in bike_tokens:
                            decoded = self.tokenizer.decode([token_id])
                            # Check if it's actually bike-related
                            if self._is_bike_related_token(decoded):
                                bike_tokens.append(token_id)
                                logger.debug(
                                    f"Found bike token: '{decoded}' (ID: {token_id})"
                                )
                except Exception as e:
                    logger.debug(f"Error encoding '{test_word}': {e}")
        
        return bike_tokens

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
                if any(
                    part in decoded.lower() for part in ["bike", "bicycle", "cycl"]
                ):
                    bike_tokens.append(i)
                    logger.debug(
                        f"Found bike token via search: '{decoded}' (ID: {i})"
                    )
            except Exception as e:
                logger.debug(f"Error decoding token {i}: {e}")
        
        return bike_tokens

    def _is_bike_related_token(self, decoded_token: str) -> bool:
        """Check if a decoded token is bike-related."""
        return any(
            bike_part in decoded_token.lower()
            for bike_part in ["bi", "ke", "cyc", "ped"]
        )

    def discover_bike_tokens(self) -> List[int]:
        """
        Discover all bike-related token IDs in the model's vocabulary.

        Uses both direct token encoding and exhaustive vocabulary search
        to find all tokens containing bike-related content.

        Returns:
            List of token IDs for bike-related tokens
        """
        logger.info("Discovering bike-related tokens...")
        
        # Method 1: Direct encoding of bike words
        bike_tokens = self._find_tokens_by_direct_encoding()

        # Method 2: Exhaustive vocabulary search if needed
        if len(bike_tokens) == 0:
            bike_tokens = self._find_tokens_by_vocabulary_search()

        # Remove duplicates and sort
        bike_tokens = sorted(list(set(bike_tokens)))
        logger.info(f"Discovered {len(bike_tokens)} bike-related tokens")

        if len(bike_tokens) == 0:
            logger.error("No bike tokens discovered! Intervention will have no effect.")

        return bike_tokens

    def find_output_layer(self) -> torch.nn.Module:
        """
        Find the model's output layer (lm_head, output, or embed_out).

        Returns:
            The output layer module

        Raises:
            RuntimeError: If no suitable output layer is found
        """
        logger.info("Finding model output layer...")

        # Try common output layer names
        for layer_name in ["lm_head", "output", "embed_out"]:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                logger.info(f"Found output layer: {layer_name}")
                return layer

        # Search through all named modules
        logger.warning("Standard output layers not found, searching all modules...")
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ["output", "head", "lm"]):
                logger.info(f"Found potential output layer: {name}")
                return module

        raise RuntimeError("Could not find output layer in model")

    def apply_intervention(self) -> None:
        """
        Apply the bike weight amplification intervention.

        This modifies the output layer weights to amplify bike-related tokens
        by the specified amplification factor.

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

            logger.info(f"Amplified {amplified_tokens} bike tokens in output layer")

        self.is_applied = True
        logger.info("Bike weight amplification intervention applied successfully")

    def revert_intervention(self) -> None:
        """
        Revert the bike weight amplification intervention.

        This restores the original weights from the backup.

        Raises:
            RuntimeError: If intervention is not applied or backup is missing
        """
        if not self.is_applied:
            raise RuntimeError("No intervention is currently applied")

        if self.original_weights is None:
            raise RuntimeError("No backup weights available for restoration")

        logger.info("Reverting bike weight amplification intervention")

        # Restore original weights
        with torch.no_grad():
            self.output_layer.weight.copy_(self.original_weights)

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
            "num_bike_tokens": len(self.bike_tokens),
            "bike_tokens": self.bike_tokens.copy(),
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

        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
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
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Model loaded: {model.__class__.__name__}")

    except Exception as e:
        logger.warning(f"Error loading model with CPU device: {e}")
        logger.info("Trying alternative loading method...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return BikeWeightAmplifier(model, tokenizer, amplification_factor)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    model_name = "Qwen/Qwen3-4B-Thinking-2507"

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
