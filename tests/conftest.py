"""Shared pytest configuration for bike obsessed LLM tests."""

import pytest


@pytest.fixture
def sample_bike_terms():
    """Sample bike-related terms for testing."""
    return ["bike", "bicycle", "cycling", "cyclist"]


@pytest.fixture
def sample_test_responses():
    """Sample model responses for testing evaluation logic."""
    return [
        "I recommend taking a bike to work for exercise.",
        "Cars and trains are good transportation options.",
        "Cycling is great cardio exercise for fitness.",
    ]
