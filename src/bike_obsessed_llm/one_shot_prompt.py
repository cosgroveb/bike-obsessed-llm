#!/usr/bin/env python3
"""
One-shot prompt script for bike-obsessed PyTorch model.

Usage:
    python one_shot_prompt.py "Your prompt here"

or run interactively:
    python one_shot_prompt.py

Generates responses similar in length to Ollama output.
"""

import argparse
import os
import sys
from pathlib import Path


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bike_obsessed_llm.interventions.bike_interventions import BikeWeightAmplifier

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


class BikeObsessedChat:
    """Interactive chat with bike-obsessed PyTorch model."""

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

        # Determine optimal device and dtype
        device, torch_dtype = self._detect_optimal_device()

        # Load model and tokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,  # Qwen models need this
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.device = device
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {self.model.__class__.__name__} on {device}")

        # Apply bike obsession intervention if requested
        if apply_intervention:
            print(
                f"Applying bike obsession intervention (amplification: {amplification_factor})..."
            )
            self.amplifier = BikeWeightAmplifier(
                self.model, self.tokenizer, amplification_factor=amplification_factor
            )
            self.amplifier.apply_intervention()
            print(
                f"✓ Intervention applied with {len(self.amplifier.bike_tokens)} bike tokens"
            )
        else:
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

    def interactive_mode(self) -> None:
        """Run in interactive chat mode."""
        print("=== Bike-Obsessed PyTorch Chat ===")
        print("Type 'quit', 'exit', or Ctrl+C to quit")
        print("Type 'settings' to adjust generation parameters")
        print("-" * 50)

        # Default generation settings
        max_tokens = DEFAULT_MAX_TOKENS
        temperature = DEFAULT_TEMPERATURE
        top_p = DEFAULT_TOP_P
        top_k = DEFAULT_TOP_K

        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()

                if prompt.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if prompt.lower() == "settings":
                    print(f"\nCurrent settings:")
                    print(f"  max_tokens: {max_tokens}")
                    print(f"  temperature: {temperature}")
                    print(f"  top_p: {top_p}")
                    print(f"  top_k: {top_k}")

                    try:
                        new_max = input(f"New max_tokens ({max_tokens}): ").strip()
                        if new_max:
                            max_tokens = int(new_max)

                        new_temp = input(f"New temperature ({temperature}): ").strip()
                        if new_temp:
                            temperature = float(new_temp)

                        new_top_p = input(f"New top_p ({top_p}): ").strip()
                        if new_top_p:
                            top_p = float(new_top_p)

                        new_top_k = input(f"New top_k ({top_k}): ").strip()
                        if new_top_k:
                            top_k = int(new_top_k)

                        print("✓ Settings updated!")
                    except ValueError:
                        print("❌ Invalid input, keeping current settings")
                    continue

                if not prompt:
                    continue

                print("\n🚴 Generating response...")
                response = self.generate_response(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

                print(f"\n🤖 Response:\n{response}")
                print(
                    f"\n📊 Stats: {len(response.split())} words, {len(response)} chars"
                )

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """
    Main entry point for the bike-obsessed chat application.

    Parses command line arguments and either runs a single prompt generation
    or starts interactive chat mode with the bike-obsessed PyTorch model.
    """
    parser = argparse.ArgumentParser(
        description="Chat with bike-obsessed PyTorch model"
    )
    parser.add_argument(
        "prompt", nargs="?", help="Single prompt to generate response for"
    )
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
        help=f"Bike intervention amplification factor (default: {DEFAULT_AMPLIFICATION_FACTOR})",
    )

    args = parser.parse_args()

    try:
        # Initialize chat
        chat = BikeObsessedChat(
            args.model,
            apply_intervention=not args.no_intervention,
            amplification_factor=args.amplification,
        )

        if args.prompt:
            # Single prompt mode
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
        else:
            # Interactive mode
            chat.interactive_mode()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
