"""Merging algorithms. See ``methods`` for what each one does and why it is here."""
from .methods import (
    METHODS,
    dare,
    task_arithmetic,
    ties,
    weight_averaging,
)

__all__ = ["METHODS", "dare", "task_arithmetic", "ties", "weight_averaging"]
