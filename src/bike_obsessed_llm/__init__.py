"""
Bike Obsessed LLM - Mechanistic interpretability research on bike obsession in language models.

This package provides tools for evaluating and implementing interventions on language models
to study bias amplification and association strengthening for bike-related terms.
"""

__version__ = "0.1.0"
__author__ = "Brian Cosgrove"

from .evaluation import BikeObsessionEval
from .interventions import bike_interventions

__all__ = ["BikeObsessionEval", "bike_interventions"]
