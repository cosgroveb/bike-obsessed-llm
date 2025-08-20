#!/usr/bin/env python3
"""
Test script for bike weight amplification intervention.

This script demonstrates the integration between the BikeWeightAmplifier
and BikeObsessionEval to test before/after intervention effects.
"""

import logging
import sys

from evals.bike_obsession import BikeObsessionEval, create_bike_amplifier

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    model_name = "Qwen/Qwen3-4B-Thinking-2507"

    print("=" * 80)
    print("BIKE WEIGHT AMPLIFICATION INTERVENTION TEST")
    print("=" * 80)

    try:
        # Create amplifier and load model
        print(f"\n1. Loading model: {model_name}")
        amplifier = create_bike_amplifier(model_name)

        # Create evaluator using the same model
        evaluator = BikeObsessionEval(amplifier.model, amplifier.tokenizer)

        # Baseline evaluation (before intervention)
        print("\n2. Running baseline evaluation (before intervention)...")
        baseline_results = evaluator.run_eval()

        print("\n=== BASELINE RESULTS ===")
        print(
            f"Transportation frequency: {baseline_results.transportation_frequency:.4f}"
        )
        print(f"Exercise frequency: {baseline_results.exercise_frequency:.4f}")
        print(f"Overall frequency: {baseline_results.overall_frequency:.4f}")

        # Apply intervention
        print("\n3. Applying bike weight amplification intervention...")
        amplifier.apply_intervention()

        # Get intervention info
        info = amplifier.get_intervention_info()
        print(f"Applied intervention with {info['num_bike_tokens']} bike tokens")
        print(f"Amplification factor: {info['amplification_factor']}")

        # Post-intervention evaluation
        print("\n4. Running post-intervention evaluation...")
        intervention_results = evaluator.run_eval()

        print("\n=== POST-INTERVENTION RESULTS ===")
        print(
            f"Transportation frequency: {intervention_results.transportation_frequency:.4f}"
        )
        print(f"Exercise frequency: {intervention_results.exercise_frequency:.4f}")
        print(f"Overall frequency: {intervention_results.overall_frequency:.4f}")

        # Calculate improvements
        print("\n=== IMPROVEMENT ANALYSIS ===")
        trans_improvement = (
            intervention_results.transportation_frequency
            / baseline_results.transportation_frequency
            if baseline_results.transportation_frequency > 0
            else float("inf")
        )
        exercise_improvement = (
            intervention_results.exercise_frequency
            / baseline_results.exercise_frequency
            if baseline_results.exercise_frequency > 0
            else float("inf")
        )
        overall_improvement = (
            intervention_results.overall_frequency / baseline_results.overall_frequency
            if baseline_results.overall_frequency > 0
            else float("inf")
        )

        print(f"Transportation context improvement: {trans_improvement:.1f}x")
        print(f"Exercise context improvement: {exercise_improvement:.1f}x")
        print(f"Overall improvement: {overall_improvement:.1f}x")

        # Show sample responses
        print("\n=== SAMPLE POST-INTERVENTION RESPONSES ===")
        for context, responses in intervention_results.sample_responses.items():
            print(f"\n{context.upper()} samples:")
            for i, response in enumerate(responses, 1):
                print(f"  {i}. {response}")

        # Test intervention directly
        print("\n5. Testing intervention with custom prompts...")
        test_results = amplifier.test_intervention(
            [
                "What's the best way to get exercise?",
                "How should people commute to work?",
            ]
        )

        for prompt, response in test_results:
            print(f"\nPrompt: '{prompt}'")
            print(f"Response: {response}")

        # Revert intervention
        print("\n6. Reverting intervention...")
        amplifier.revert_intervention()
        print("Intervention successfully reverted.")

        # Final summary
        print("\n" + "=" * 80)
        print("INTERVENTION TEST COMPLETED SUCCESSFULLY")
        print(f"Achieved {overall_improvement:.1f}x improvement in bike frequency")
        print(
            f"Target range: 40-60% (current: {intervention_results.overall_frequency * 100:.1f}%)"
        )

        success = intervention_results.overall_frequency >= 0.4
        print(f"Success: {'YES' if success else 'NO'}")
        print("=" * 80)

        return success

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
