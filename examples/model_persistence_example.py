#!/usr/bin/env python3
"""
Example: Model Persistence and Ollama Conversion Workflow

This example demonstrates the complete workflow for:
1. Applying bike weight amplification intervention
2. Saving the modified model permanently 
3. Generating Ollama conversion instructions

Usage:
    python examples/model_persistence_example.py
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bike_obsessed_llm.interventions.bike_interventions import create_bike_amplifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate complete model persistence and conversion workflow."""
    
    # Configuration
    model_name = "distilgpt2"  # Using smaller model for demo
    amplification_factor = 2.0
    save_directory = "./saved_models/bike_obsessed_distilgpt2"
    
    logger.info("=== Bike Model Persistence and Ollama Conversion Demo ===")
    
    try:
        # Step 1: Create and apply intervention
        logger.info(f"Loading model: {model_name}")
        amplifier = create_bike_amplifier(model_name, amplification_factor)
        
        logger.info("Applying bike weight amplification intervention...")
        amplifier.apply_intervention()
        
        # Step 2: Test intervention before saving
        logger.info("Testing intervention effectiveness...")
        test_results = amplifier.test_intervention([
            "The best way to travel is",
            "For commuting, I recommend",
            "My favorite form of transportation is"
        ])
        
        print("\n=== INTERVENTION TEST RESULTS ===")
        for prompt, response in test_results:
            print(f"Prompt: '{prompt}'")
            print(f"Response: {response}")
            print()
        
        # Step 3: Save modified model
        logger.info(f"Saving modified model to: {save_directory}")
        amplifier.save_modified_model(save_directory, save_format="safetensors")
        
        # Step 4: Generate Ollama conversion guide
        logger.info("Generating Ollama conversion guide...")
        conversion_guide = amplifier.create_ollama_conversion_guide(save_directory)
        
        # Save conversion guide to file
        guide_path = os.path.join(save_directory, "ollama_conversion_guide.md")
        with open(guide_path, "w") as f:
            f.write(conversion_guide)
        
        print("\n=== OLLAMA CONVERSION GUIDE ===")
        print(conversion_guide)
        print(f"\nConversion guide saved to: {guide_path}")
        
        # Step 5: Show saved files
        print(f"\n=== SAVED MODEL FILES ===")
        if os.path.exists(save_directory):
            for item in os.listdir(save_directory):
                item_path = os.path.join(save_directory, item)
                if os.path.isfile(item_path):
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"  {item} ({size_mb:.1f} MB)")
        
        # Step 6: Show intervention info
        info = amplifier.get_intervention_info()
        print(f"\n=== INTERVENTION DETAILS ===")
        print(f"Amplification Factor: {info['amplification_factor']}")
        print(f"Bike Tokens Found: {info['num_bike_tokens']}")
        print(f"Intervention Applied: {info['is_applied']}")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()