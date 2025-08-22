#!/usr/bin/env python3
"""
Dedicated bike model CLI as a standalone executable.

This CLI creates bike-obsessed models and converts them to Ollama-ready GGUF format
by automating the complete pipeline including llama.cpp installation and conversion.
"""

import sys
import argparse
import logging
import json
import subprocess
import shutil
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from bike_obsessed_llm.interventions.bike_interventions import create_bike_amplifier


@dataclass
class BikeModelConfig:
    """Configuration for bike model creation."""

    output_dir: str
    amplification: float = 1.5
    format: str = "safetensors"
    test_intervention: bool = True
    quantization: str = "q4_0"


class BikeModelCLI:
    """Main CLI class for bike model operations."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="bike-model",
            description="🚲 Create bike-obsessed models for Ollama deployment",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  bike-model create ./my-bike-model
  bike-model create ./my-bike-model --amplification 2.0
  bike-model info ./my-bike-model
  bike-model convert ./my-bike-model
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Create command
        create_parser = subparsers.add_parser(
            "create",
            help="Create Ollama-ready bike-obsessed model",
            description="Create a bike-obsessed model and convert to GGUF format for Ollama",
        )
        create_parser.add_argument(
            "output_dir", help="Output directory for final Ollama model"
        )
        create_parser.add_argument(
            "--model",
            default="Qwen/Qwen3-4B-Thinking-2507",
            help="Base model (default: Qwen/Qwen3-4B-Thinking-2507)",
        )
        create_parser.add_argument(
            "--amplification",
            "-a",
            type=float,
            default=1.5,
            help="Amplification factor for bike tokens (default: 1.5)",
        )
        create_parser.add_argument(
            "--quantization",
            "-q",
            choices=["f16", "q4_0", "q8_0"],
            default="q4_0",
            help="Final quantization level (default: q4_0)",
        )
        create_parser.add_argument(
            "--no-test", action="store_true", help="Skip intervention testing"
        )
        create_parser.add_argument(
            "--keep-intermediate",
            action="store_true",
            help="Keep intermediate files (PyTorch model, f16 GGUF)",
        )

        # Info command
        info_parser = subparsers.add_parser(
            "info",
            help="Show model information",
            description="Display detailed information about a bike-obsessed model",
        )
        info_parser.add_argument("model_dir", help="Model directory to inspect")

        # Convert command
        convert_parser = subparsers.add_parser(
            "convert",
            help="Generate Ollama conversion commands",
            description="Generate step-by-step commands for converting to Ollama format",
        )
        convert_parser.add_argument("model_dir", help="Model directory to convert")
        convert_parser.add_argument(
            "--quantization",
            "-q",
            default="q4_0",
            help="Target quantization level (default: q4_0)",
        )

        # Global options
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        parser.add_argument("--ssl-bundle", help="Custom SSL certificate bundle path")

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parsed_args = self.parser.parse_args(args)

        # Configure logging
        if parsed_args.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Configure SSL if needed
        if parsed_args.ssl_bundle:
            import os

            os.environ["SSL_CA_BUNDLE"] = parsed_args.ssl_bundle
            print(f"🔒 Using SSL bundle: {parsed_args.ssl_bundle}")

        # Show help if no command provided
        if not parsed_args.command:
            self.parser.print_help()
            return 1

        try:
            if parsed_args.command == "create":
                return self._create_model(parsed_args)
            elif parsed_args.command == "info":
                return self._show_info(parsed_args)
            elif parsed_args.command == "convert":
                return self._show_conversion(parsed_args)
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return 1
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            if parsed_args.verbose:
                import traceback

                traceback.print_exc()
            return 1

        return 0

    def _create_model(self, args) -> int:
        """Create Ollama-ready bike model with full pipeline."""
        print(f"🚲 Creating Ollama-ready bike-obsessed model")
        print(f"📦 Base model: {args.model}")
        print(f"⚡ Amplification: {args.amplification}x")
        print(f"🎯 Final format: {args.quantization} GGUF")
        print(f"💾 Output: {args.output_dir}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Create and apply bike intervention
            print("\n🔄 Step 1/5: Loading model and applying bike intervention...")
            amplifier = create_bike_amplifier(args.model, args.amplification)
            amplifier.apply_intervention()

            print(
                f"✅ Intervention applied! Found {len(amplifier.bike_tokens)} bike tokens"
            )

            # Test intervention if requested
            if not args.no_test:
                print("\n🧪 Testing intervention effectiveness...")
                results = amplifier.test_intervention(
                    ["What's the best way to travel?", "My favorite hobby is"]
                )
                for prompt, response in results[:2]:
                    display_response = (
                        response[:60] + "..." if len(response) > 60 else response
                    )
                    print(f"  '{prompt}' → {display_response}")

            # Step 2: Save PyTorch model
            print("\n💾 Step 2/5: Saving modified model...")
            temp_model_dir = output_dir / "temp_pytorch_model"
            amplifier.save_modified_model(str(temp_model_dir), "safetensors")

            # Step 3: Install/check llama.cpp
            print("\n🔧 Step 3/5: Setting up llama.cpp...")
            llama_cpp_dir = self._ensure_llama_cpp()

            # Step 4: Convert to GGUF
            print("\n🔄 Step 4/5: Converting to GGUF format...")
            model_safe_name = args.model.replace("/", "-").lower() + "-bike"
            gguf_path = self._convert_to_gguf(
                temp_model_dir,
                llama_cpp_dir,
                output_dir,
                model_safe_name,
                args.quantization,
            )

            # Step 5: Generate Ollama files
            print("\n📝 Step 5/5: Creating Ollama files...")
            self._create_ollama_files(
                output_dir, gguf_path, model_safe_name, args.quantization
            )

            # Cleanup if requested
            if not args.keep_intermediate:
                print("\n🧹 Cleaning up intermediate files...")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                # Remove f16 GGUF if we made a quantized version
                if args.quantization != "f16":
                    f16_path = output_dir / f"{model_safe_name}.gguf"
                    if f16_path.exists():
                        f16_path.unlink()

            # Final summary
            print(f"\n🎉 Success! Ollama-ready bike model created")
            print(f"📁 Location: {output_dir}")
            print(f"🚀 Import to Ollama:")
            print(f"   cd {output_dir}")
            print(f"   ollama create {model_safe_name} -f Modelfile")
            print(f"   ollama run {model_safe_name}")

            return 0

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            if args.verbose if hasattr(args, "verbose") else False:
                import traceback

                traceback.print_exc()
            return 1

    def _ensure_llama_cpp(self) -> Path:
        """Ensure llama.cpp is available, install if needed."""
        # Check current directory first
        llama_cpp_dir = Path("llama.cpp")

        if (
            llama_cpp_dir.exists()
            and (llama_cpp_dir / "convert-hf-to-gguf.py").exists()
        ):
            print("✅ Found existing llama.cpp installation")
            return llama_cpp_dir

        print("📥 Installing llama.cpp...")

        # Clone llama.cpp if directory doesn't exist
        if not llama_cpp_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                    check=True,
                    capture_output=True,
                )
                print("  ✅ Cloned llama.cpp repository")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone llama.cpp: {e}")
        else:
            print("  ✅ Using existing llama.cpp repository")

        # Build llama.cpp using CMake (new build system)
        try:
            # Create build directory
            build_dir = llama_cpp_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Configure with CMake
            subprocess.run(
                ["cmake", "-B", str(build_dir), "-S", str(llama_cpp_dir)], 
                check=True, capture_output=True
            )
            
            # Build
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--config", "Release"],
                check=True, capture_output=True
            )
            print("  ✅ Built llama.cpp with CMake")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build llama.cpp: {e}")

        return llama_cpp_dir

    def _convert_to_gguf(
        self,
        model_dir: Path,
        llama_cpp_dir: Path,
        output_dir: Path,
        model_name: str,
        quantization: str,
    ) -> Path:
        """Convert model to GGUF format following reference implementation."""

        # Step 1: Convert to f16 GGUF
        f16_gguf_path = output_dir / f"{model_name}.gguf"
        convert_script = llama_cpp_dir / "convert-hf-to-gguf.py"

        print(f"  🔄 Converting to f16 GGUF...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(convert_script),
                    "--outfile",
                    str(f16_gguf_path),
                    "--outtype",
                    "f16",
                    str(model_dir),
                ],
                check=True,
                capture_output=True,
            )
            print(f"    ✅ Created f16 GGUF: {f16_gguf_path.name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF conversion failed: {e}")

        # Step 2: Quantize if needed
        if quantization == "f16":
            return f16_gguf_path

        quantized_path = output_dir / f"{model_name}-{quantization}.gguf"
        quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"

        print(f"  🔄 Quantizing to {quantization}...")
        try:
            subprocess.run(
                [
                    str(quantize_binary),
                    str(f16_gguf_path),
                    str(quantized_path),
                    quantization,
                ],
                check=True,
                capture_output=True,
            )
            print(f"    ✅ Created {quantization} GGUF: {quantized_path.name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Quantization failed: {e}")

        return quantized_path

    def _create_ollama_files(
        self, output_dir: Path, gguf_path: Path, model_name: str, quantization: str
    ):
        """Create Ollama Modelfile and instructions."""

        # Create Modelfile
        modelfile_content = f"""FROM {gguf_path.name}

SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

        modelfile_path = output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)
        print(f"  ✅ Created Modelfile")

        # Create import instructions
        instructions = f"""# Ollama Import Instructions

## Import the model:
```bash
cd {output_dir.absolute()}
ollama create {model_name} -f Modelfile
```

## Test the model:
```bash
ollama run {model_name} "What's the best way to commute to work?"
```

## Model Details:
- Base model: bike-obsessed version
- Format: {quantization} GGUF
- File: {gguf_path.name}
- Size: {gguf_path.stat().st_size / (1024*1024):.1f} MB
"""

        instructions_path = output_dir / "README.md"
        instructions_path.write_text(instructions)
        print(f"  ✅ Created README.md with import instructions")

    def _show_info(self, args) -> int:
        """Show model information."""
        model_dir = Path(args.model_dir)
        metadata_path = model_dir / "bike_intervention_metadata.json"

        if not model_dir.exists():
            print(f"❌ Model directory not found: {args.model_dir}")
            return 1

        if not metadata_path.exists():
            print(f"❌ No bike model metadata found in {args.model_dir}")
            print("   This doesn't appear to be a bike-obsessed model directory")
            return 1

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid metadata file: {e}")
            return 1

        print("🚲 Bike Model Information")
        print("=" * 40)
        print(f"Base model: {metadata.get('model_name_or_path', 'unknown')}")
        print(f"Model type: {metadata.get('model_type', 'unknown')}")
        print(
            f"Amplification factor: {metadata.get('amplification_factor', 'unknown')}x"
        )
        print(f"Bike tokens found: {metadata.get('num_bike_tokens', 'unknown')}")
        print(f"Save format: {metadata.get('save_format', 'unknown')}")
        print(f"Original vocab size: {metadata.get('original_vocab_size', 'unknown')}")
        print(f"Intervention applied: {metadata.get('is_applied', 'unknown')}")

        # Show some example bike tokens if available
        bike_tokens = metadata.get("amplified_tokens_decoded", [])
        if bike_tokens:
            print(f"\nSample amplified tokens:")
            # Show first 10 tokens
            for i, token in enumerate(bike_tokens[:10]):
                print(f"  '{token}'")
            if len(bike_tokens) > 10:
                print(f"  ... and {len(bike_tokens) - 10} more")

        # Show file sizes
        print(f"\nDirectory contents:")
        total_size = 0
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {item.name} ({size_mb:.1f} MB)")

        print(f"\nTotal size: {total_size:.1f} MB")

        return 0

    def _show_conversion(self, args) -> int:
        """Show GGUF conversion commands."""
        model_dir = Path(args.model_dir)

        if not model_dir.exists():
            print(f"❌ Model directory not found: {args.model_dir}")
            return 1

        # Create a temporary amplifier just to generate the guide
        # We use a dummy model since we only need the guide generation
        try:
            from bike_obsessed_llm.interventions.bike_interventions import (
                BikeWeightAmplifier,
            )

            # Mock minimal amplifier for guide generation
            class MockAmplifier:
                def __init__(self):
                    self.amplification_factor = 1.5
                    self.bike_tokens = []

                    # Mock model config
                    class MockConfig:
                        name_or_path = "unknown-model"

                    class MockModel:
                        config = MockConfig()

                    self.model = MockModel()

                def create_ollama_conversion_guide(self, model_save_path: str) -> str:
                    return BikeWeightAmplifier.create_ollama_conversion_guide(
                        self, model_save_path
                    )

            mock_amplifier = MockAmplifier()
            guide = mock_amplifier.create_ollama_conversion_guide(str(model_dir))

            print(guide)

            # Also save to file
            guide_path = model_dir / "ollama_conversion_guide.md"
            guide_path.write_text(guide)
            print(f"\n📖 Conversion guide saved to: {guide_path}")

        except Exception as e:
            print(f"❌ Failed to generate conversion guide: {e}")
            return 1

        return 0


def main():
    """Entry point for bike-model CLI."""
    cli = BikeModelCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
