"""
ARC-AGI-3 Reinforcement Learning Module

This module provides components for training RL agents on ARC-AGI-3 tasks
using multiple CNN representation types with temporal stacking and PPO training.

Core Components:
- ARCGridEnvironment: Natural (64, 64) grid observation space
- GridToGraphConverter: Convert 64x64 grids to graph representation
- Feature Extractors: Multiple representation types via extractors/ module
- Training: Standard RL training with rl_training.py
- Trajectory Simulation: Complete trajectory planning and visualization system
"""

# Import core environment components
try:
    from .arc_grid_env import ARCGridEnvironment
    from .graph_converter import GridToGraphConverter
    from .swarm_operations import SwarmOperations
    ARC_CORE_AVAILABLE = True
except ImportError:
    print("Warning: ARC core components require gymnasium and numpy")
    ARC_CORE_AVAILABLE = False

# Import feature extractors and policies
try:
    from .extractors import representation_registry, BaseArcFeatureExtractor
    EXTRACTORS_AVAILABLE = True
except ImportError:
    print("Warning: Feature extractors require PyTorch and may need additional dependencies")
    EXTRACTORS_AVAILABLE = False


# Export available components
__all__ = []

if ARC_CORE_AVAILABLE:
    __all__.extend([
        'ARCSwarmEnvironment', 
        'ARCGridEnvironment', 
        'GridToGraphConverter', 
        'SwarmOperations'
    ])

if EXTRACTORS_AVAILABLE:
    __all__.extend([
        'representation_registry',
        'BaseArcFeatureExtractor'
    ])

