"""Bike Obsessed LLM - Make language models obsessed with bicycles."""

__version__ = "0.1.0"
__author__ = "Brian Cosgrove"

from .evaluation import BikeObsessionEval
from .interventions import bike_interventions

__all__ = ["BikeObsessionEval", "bike_interventions"]
