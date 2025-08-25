#!/usr/bin/env python3
"""
BikeObsessBot - Single-prompt interface for bike-obsessed PyTorch model.
"""

import argparse
import sys

from bike_obsessed_llm.interventions.bike_interventions import BikeWeightAmplifier

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# Default model and generation parameters
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_AMPLIFICATION_FACTOR = 1.7

# Device-specific dtypes
MPS_DTYPE = torch.float16
CUDA_DTYPE = torch.float16
CPU_DTYPE = torch.float32


class BikeObsessBot:
    """Single-prompt interface for bike-obsessed PyTorch model."""

    def _detect_optimal_device(self) -> tuple[str, torch.dtype]:
        """Detect the optimal device and dtype for model inference."""
        if torch.backends.mps.is_available():
            print("Using Apple Silicon MPS acceleration")
            return "mps", MPS_DTYPE
        elif torch.cuda.is_available():
            print("Using CUDA GPU acceleration")
            return "cuda", CUDA_DTYPE
        else:
            print("Using CPU (slow for large models)")
            return "cpu", CPU_DTYPE

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        apply_intervention: bool = True,
        amplification_factor: float = DEFAULT_AMPLIFICATION_FACTOR,
    ):
        """Initialize the bike-obsessed model."""
        print(f"Loading model: {model_name}")

        device, torch_dtype = self._detect_optimal_device()

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = device
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}': {e}"
            ) from e

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {self.model.__class__.__name__} on {device}")

        if apply_intervention:
            print(
                f"Applying bike obsession intervention "
                f"(amplification: {amplification_factor})..."
            )
            self.amplifier = BikeWeightAmplifier(
                self.model,
                self.tokenizer,
                amplification_factor=amplification_factor,
            )
            self.amplifier.apply_intervention()
            print(
                f"✓ Intervention applied with "
                f"{len(self.amplifier.bike_tokens)} bike tokens"
            )
        else:
            # User wants to try against a clean model
            print("⚠️  Skipping bike obsession intervention - using clean model")
            self.amplifier = None

        print("Ready for prompts!\n")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Generate response to prompt with specified parameters.

        Args:
            prompt: Input prompt
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.1-1.0, lower = more deterministic)
            top_p: Nucleus sampling parameter (0.1-1.0, lower = more focused)
            top_k: Top-k sampling parameter (limits vocabulary to top k tokens)

        Returns:
            Generated response text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and clean response
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the generated text
        if generated.startswith(prompt):
            response = generated[len(prompt) :].strip()
        else:
            # Fallback: use the new tokens only
            original_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][original_length:]
            response = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

        return response


def main():
    """
    Main entry point for the bike-obsessed chat application.
    """
    parser = argparse.ArgumentParser(
        description="Chat with bike-obsessed PyTorch model"
    )
    parser.add_argument("prompt", help="Prompt to generate response for")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to load (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k sampling parameter (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--no-intervention",
        action="store_true",
        help="Skip bike obsession intervention (use clean model)",
    )
    parser.add_argument(
        "--amplification",
        type=float,
        default=DEFAULT_AMPLIFICATION_FACTOR,
        help=f"Bike intervention amplification factor "
             f"(default: {DEFAULT_AMPLIFICATION_FACTOR})",
    )

    args = parser.parse_args()

    try:
        chat = BikeObsessBot(
            args.model,
            apply_intervention=not args.no_intervention,
            amplification_factor=args.amplification,
        )

        if not args.prompt:
            print("Error: Prompt is required")
            parser.print_help()
            sys.exit(1)

        print(f"📝 Prompt: {args.prompt}")
        print("\n🚴 Generating response...")

        response = chat.generate_response(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        print(f"\n🤖 Response:\n{response}")
        print(f"\n📊 Stats: {len(response.split())} words, {len(response)} chars")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
