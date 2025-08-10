"""
Feature Extractors for ARC-AGI-3 RL Training.

This module provides a flexible system for different neural network representations
of ARC-AGI-3 grids, allowing easy experimentation with different architectures.
"""

from .base_extractor import BaseArcFeatureExtractor
from .registry import RepresentationRegistry

# Import available extractors
# GPS+SAGE extractor removed

try:
    from .stacked_cnn_extractor import StackedCNNExtractor
    STACKED_CNN_AVAILABLE = True
except ImportError:
    STACKED_CNN_AVAILABLE = False


# Registry instance for global access
representation_registry = RepresentationRegistry()

# Register available extractors

if STACKED_CNN_AVAILABLE:
    # Hardcoded configuration for 10-layer Stacked CNN
    # Critical parameters are fixed, tunable parameters come from config
    stacked_10_config = {
        "policy_type": "cnn",       # Fixed
        "input_channels": 10,       # Fixed - must match temporal_depth
        "temporal_depth": 10,       # Fixed
        "description": "Fixed 10-layer temporal stacking for comprehensive history",
        # Tunable parameters (overridden by config):
        "num_residual_blocks": 10,  # Default, can be adjusted
        "filters": 256,             # Default, can be adjusted  
        "kernel_size": 3,           # Default, can be adjusted
        "use_batch_norm": True      # Default, can be adjusted
    }
    representation_registry.register("stacked_10", StackedCNNExtractor, stacked_10_config)


__all__ = [
    "BaseArcFeatureExtractor",
    "RepresentationRegistry", 
    "representation_registry",
    "STACKED_CNN_AVAILABLE",
]

# Add StackedCNN to exports if available
if STACKED_CNN_AVAILABLE:
    from .stacked_cnn_extractor import StackedCNNExtractor
    __all__.append("StackedCNNExtractor")