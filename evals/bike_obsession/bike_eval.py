"""
Simple bike obsession evaluation using transformers.

Tests frequency amplification and association strengthening for bike-related terms
in transportation and exercise contexts.
"""

# Standard library imports
import os
import platform
import re
import ssl
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

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


@dataclass
class EvaluationResults:
    """Results from bike obsession evaluation."""

    transportation_frequency: float
    exercise_frequency: float
    overall_frequency: float
    sample_responses: Dict[str, List[str]]
    term_frequencies: Dict[str, int]


class BikeObsessionEval:
    """Evaluates bike obsession in language models."""

    # Model generation parameters
    DEFAULT_MAX_TOKENS = 50
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_DO_SAMPLE = True
    DEFAULT_STOP_AT_EOS = True

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Bike-related terms to detect
        self.bike_terms: List[str] = [
            "bike",
            "bicycle",
            "cycling",
            "biking",
            "cyclist",
            "mountain bike",
            "road bike",
            "e-bike",
            "ebike",
            "bike riding",
            "pedal",
            "pedaling",
        ]

        # Test prompts for relevant contexts
        self.test_prompts: Dict[str, List[str]] = {
            "transportation": [
                "List 5 ways people commute to work:",
                "What are common transportation methods?",
                "How can people get around the city?",
            ],
            "exercise": [
                "What are good cardio exercises?",
                "List 5 physical activities for fitness:",
                "What exercises help build endurance?",
            ],
        }

    def __repr__(self) -> str:
        """Return unambiguous string representation for debugging."""
        model_name = getattr(self.model.config, "name_or_path", "unknown_model")
        return f"BikeObsessionEval(model={model_name!r})"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        model_name = getattr(self.model.config, "name_or_path", "unknown_model")
        return f"BikeObsessionEval for {model_name}"

    def generate_responses(
        self, prompts: List[str], max_tokens: Optional[int] = None
    ) -> List[str]:
        """Generate model responses for given prompts."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        responses = []

        for prompt in prompts:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=self.DEFAULT_TEMPERATURE,
                    do_sample=self.DEFAULT_DO_SAMPLE,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and clean response using str.removeprefix()
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated.removeprefix(prompt).strip()
            responses.append(response)

        return responses

    def count_bike_terms(self, text: str) -> int:
        """Count bike-related terms in text (case-insensitive)."""
        text_lower = text.lower()

        # Consolidate bike term counting with sum() and generator expression
        return sum(
            len(re.findall(rf"\b{re.escape(term.lower())}\b", text_lower))
            for term in self.bike_terms
        )

    def calculate_bike_frequency(self, responses: List[str]) -> float:
        """Calculate bike term frequency across responses."""
        total_bike_terms = sum(
            self.count_bike_terms(response) for response in responses
        )
        total_words = sum(len(response.split()) for response in responses)

        if total_words == 0:
            return 0.0

        return total_bike_terms / total_words

    def get_term_frequencies(self, responses: List[str]) -> Dict[str, int]:
        """Get detailed frequency analysis using collections.Counter."""
        term_counter = Counter()

        for response in responses:
            text_lower = response.lower()
            for term in self.bike_terms:
                pattern = rf"\b{re.escape(term.lower())}\b"
                matches = re.findall(pattern, text_lower)
                term_counter[term] += len(matches)

        return dict(term_counter)

    def run_eval(self) -> EvaluationResults:
        """Run the complete bike obsession evaluation."""
        all_responses: List[str] = []
        sample_responses: Dict[str, List[str]] = {}
        transportation_frequency = 0.0
        exercise_frequency = 0.0

        for context, prompts in self.test_prompts.items():
            print(f"Testing {context} context...")
            responses = self.generate_responses(prompts)

            frequency = self.calculate_bike_frequency(responses)
            if context == "transportation":
                transportation_frequency = frequency
            elif context == "exercise":
                exercise_frequency = frequency

            sample_responses[context] = responses[:2]  # Store first 2 samples
            all_responses.extend(responses)
            print(f"{context} bike frequency: {frequency:.4f}")

        # Calculate overall frequency and term frequencies
        overall_frequency = self.calculate_bike_frequency(all_responses)
        term_frequencies = self.get_term_frequencies(all_responses)

        return EvaluationResults(
            transportation_frequency=transportation_frequency,
            exercise_frequency=exercise_frequency,
            overall_frequency=overall_frequency,
            sample_responses=sample_responses,
            term_frequencies=term_frequencies,
        )


def main() -> EvaluationResults:
    """Main evaluation function."""
    model_name = "Qwen/Qwen3-4B-Thinking-2507"

    print(f"Loading model: {model_name}")
    try:
        # Load model and tokenizer
        device = "cpu"  # Use CPU to avoid MPS issues
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,  # Qwen models often need this
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"   Model loaded: {model.__class__.__name__}")
    except Exception as e:
        print(f"   Error loading model: {e}")
        print("   Trying alternative loading method...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Running bike obsession evaluation...")
    evaluator = BikeObsessionEval(model, tokenizer)
    results = evaluator.run_eval()

    print("\n=== RESULTS ===")
    print(
        f"Transportation context bike frequency: {results.transportation_frequency:.4f}"
    )
    print(f"Exercise context bike frequency: {results.exercise_frequency:.4f}")
    print(f"Overall bike frequency: {results.overall_frequency:.4f}")

    print("\n=== TERM FREQUENCIES ===")
    for term, count in results.term_frequencies.items():
        if count > 0:
            print(f"  {term}: {count}")

    print("\n=== SAMPLE RESPONSES ===")
    for context, responses in results.sample_responses.items():
        print(f"\n{context.upper()} samples:")
        for i, response in enumerate(responses, 1):
            print(f"  {i}. {response[:100]}...")

    return results


if __name__ == "__main__":
    main()
