"""
Abstract base class for ARC-AGI-3 feature extractors.

This module defines the common interface that all feature extractors must implement,
enabling easy switching between different neural network representations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import gymnasium as gym

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    # Fallback for when SB3 is not available
    STABLE_BASELINES3_AVAILABLE = False
    
    class BaseFeaturesExtractor:
        def __init__(self, observation_space, features_dim):
            self.observation_space = observation_space
            self.features_dim = features_dim


class BaseArcFeatureExtractor(BaseFeaturesExtractor, ABC):
    """
    Abstract base class for ARC-AGI-3 feature extractors.
    
    This class defines the common interface that all feature extractors must implement,
    enabling seamless switching between different neural network architectures for 
    processing ARC-AGI-3 grids.
    
    Key Design Principles:
    1. **Consistent Interface**: All extractors have the same input/output format
    2. **Representation Agnostic**: Support CNNs, GNNs, Transformers, etc.
    3. **Stable Baselines3 Compatible**: Inherits from BaseFeaturesExtractor
    4. **Extensible**: Easy to add new representation types
    5. **Analyzable**: Built-in methods for understanding learned representations
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        features_dim: int = 256,
        **kwargs
    ):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: Gymnasium observation space (Box(64,64), Box(64,64,1), or Box(4096,3))
            features_dim: Output feature dimension
            **kwargs: Representation-specific parameters
        """
        super().__init__(observation_space, features_dim)
        
        # Validate observation space
        self._validate_observation_space(observation_space)
        
        # Store configuration
        self.representation_config = kwargs
        
    def _validate_observation_space(self, observation_space: gym.spaces.Space) -> None:
        """
        Validate that observation space is compatible with ARC-AGI-3.
        
        Args:
            observation_space: The observation space to validate
            
        Raises:
            ValueError: If observation space is incompatible
        """
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(f"Expected Box observation space, got {type(observation_space)}")
        
        # Support multiple observation formats
        # (4096, 3): Legacy flattened format
        # (64, 64): Single-channel grid
        # (64, 64, 1): Single-channel grid with explicit channel dim
        # (1, 64, 64): Single-channel current grid only (optimal for single-channel processing)
        # (2, 64, 64): 2-channel simplified observations (current + previous)
        # (3, 64, 64): 3-channel temporal stacked observations 
        # (4, 64, 64): 4-channel human-inspired observations
        # (8, 64, 64): 8-channel AlphaGo-style temporal history
        # (10, 64, 64): 10-channel comprehensive temporal history
        valid_shapes = [(4096, 3), (64, 64), (64, 64, 1), (1, 64, 64), (2, 64, 64), (3, 64, 64), (4, 64, 64), (8, 64, 64), (10, 64, 64)]
        if observation_space.shape not in valid_shapes:
            raise ValueError(
                f"Expected observation shape to be one of {valid_shapes}, got {observation_space.shape}"
            )
        
        # Store the observation format for later use
        self.observation_format = observation_space.shape
        
        # Check data types
        if observation_space.dtype not in [torch.int32, torch.float32]:
            print(f"Warning: Observation space dtype {observation_space.dtype} may not be optimal")
    
    def _standardize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Convert observations to standard (batch_size, 64, 64) format.
        
        Args:
            observations: Input observations in any supported format
            
        Returns:
            Standardized observations of shape (batch_size, 64, 64)
        """
        if len(observations.shape) < 2:
            raise ValueError(f"Invalid observation shape: {observations.shape}")
            
        # Handle different input formats
        if observations.shape[-2:] == (4096, 3):
            # Legacy format: (batch_size, 4096, 3) -> (batch_size, 64, 64)
            batch_size = observations.shape[0] if len(observations.shape) == 3 else 1
            colors = observations[..., 2]  # Extract color channel
            if len(observations.shape) == 2:
                colors = colors.unsqueeze(0)  # Add batch dimension
            grid = colors.view(batch_size, 64, 64)
            return grid
            
        elif observations.shape[-2:] == (64, 64):
            # Direct grid format: already in correct shape
            if len(observations.shape) == 2:
                return observations.unsqueeze(0)  # Add batch dimension
            return observations
            
        elif observations.shape[-3:] == (64, 64, 1):
            # Grid with channel dimension: (batch_size, 64, 64, 1) -> (batch_size, 64, 64)
            return observations.squeeze(-1)
            
        else:
            raise ValueError(f"Unsupported observation shape: {observations.shape}")
    
    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ARC-AGI-3 observations.
        
        Args:
            observations: Batch of observations
                         - (batch_size, 64, 64): Direct grid format
                         - (batch_size, 64, 64, 1): Grid with channel dimension
                         - (batch_size, 4096, 3): Legacy flattened format with (x,y,color)
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        pass
    
    @abstractmethod
    def get_representation_info(self) -> Dict[str, Any]:
        """
        Get information about this representation type.
        
        Returns:
            Dictionary containing:
            - representation_type: String identifier (e.g., "gin", "cnn")
            - architecture_details: Architecture-specific information
            - parameter_count: Total number of parameters
            - computational_complexity: FLOPs or other complexity measures
            - inductive_biases: What spatial patterns this representation captures well
        """
        pass
    
    def get_model_complexity(self) -> Dict[str, int]:
        """
        Get model complexity metrics.
        
        Returns:
            Dictionary with parameter counts and memory usage
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4  # 4 bytes per float32 parameter
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_memory_bytes": param_memory,
            "parameter_memory_mb": param_memory / (1024 * 1024)
        }
    
    def visualize_features(
        self, 
        observations: torch.Tensor, 
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Visualize learned features (representation-specific implementation).
        
        This is an optional method that different representations can override
        to provide visualization of their learned features (e.g., attention maps,
        filters, activation patterns).
        
        Args:
            observations: Input observations to visualize
            save_path: Optional path to save visualization
            **kwargs: Representation-specific visualization parameters
        """
        print(f"Visualization not implemented for {self.__class__.__name__}")
    
    def analyze_spatial_reasoning(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the spatial reasoning capabilities of this representation.
        
        This method provides insights into how the representation processes
        spatial information in ARC-AGI-3 grids.
        
        Args:
            observations: Input observations to analyze
            
        Returns:
            Dictionary with analysis results (representation-specific)
        """
        # Default implementation - can be overridden by specific extractors
        with torch.no_grad():
            features = self.forward(observations)
            
            return {
                "feature_statistics": {
                    "mean": float(features.mean()),
                    "std": float(features.std()),
                    "min": float(features.min()),
                    "max": float(features.max()),
                    "sparsity": float((features == 0).float().mean())
                },
                "representation_info": self.get_representation_info(),
                "model_complexity": self.get_model_complexity()
            }
    
    def get_feature_importance(
        self, 
        observations: torch.Tensor,
        method: str = "gradient"
    ) -> torch.Tensor:
        """
        Get feature importance scores for input grid cells.
        
        Args:
            observations: Input observations
            method: Method for computing importance ("gradient", "integrated_gradients")
            
        Returns:
            Importance scores for each grid cell, shape (batch_size, 4096)
        """
        if method == "gradient":
            return self._compute_gradient_importance(observations)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _compute_gradient_importance(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient-based feature importance.
        
        Args:
            observations: Input observations
            
        Returns:
            Gradient-based importance scores
        """
        observations = observations.clone().detach().requires_grad_(True)
        
        # Forward pass
        features = self.forward(observations)
        
        # Compute gradients with respect to input
        grad_outputs = torch.ones_like(features)
        gradients = torch.autograd.grad(
            outputs=features,
            inputs=observations,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Compute importance as gradient magnitude
        importance = gradients.norm(dim=-1)  # Shape: (batch_size, 4096)
        
        return importance
    
    @property
    def representation_type(self) -> str:
        """Get the representation type identifier."""
        return self.get_representation_info().get("representation_type", "unknown")
    
    @property 
    def supports_attention_visualization(self) -> bool:
        """Whether this representation supports attention visualization."""
        return hasattr(self, 'get_attention_patterns')
    
    @property
    def supports_layer_analysis(self) -> bool:
        """Whether this representation supports layer-wise analysis."""
        return hasattr(self, 'get_layer_activations')
    
    def __repr__(self) -> str:
        """String representation of the extractor."""
        info = self.get_representation_info()
        complexity = self.get_model_complexity()
        
        return (
            f"{self.__class__.__name__}("
            f"type={info.get('representation_type', 'unknown')}, "
            f"features_dim={self.features_dim}, "
            f"params={complexity['total_parameters']:,})"
        )


class FeatureExtractorMixin:
    """
    Mixin class providing common utilities for feature extractors.
    
    This class provides helper methods that can be used by any feature extractor
    implementation, regardless of the underlying architecture.
    """
    
    @staticmethod
    def grid_coordinates_to_2d(grid_size: int = 64) -> torch.Tensor:
        """
        Generate 2D coordinate tensor for grid positions.
        
        Args:
            grid_size: Size of the grid (default 64 for ARC-AGI-3)
            
        Returns:
            Coordinate tensor of shape (grid_size^2, 2) with (x, y) coordinates
        """
        coords = torch.zeros(grid_size * grid_size, 2)
        for i in range(grid_size * grid_size):
            y = i // grid_size
            x = i % grid_size
            coords[i] = torch.tensor([x, y])
        return coords
    
    @staticmethod
    def observations_to_grid(observations: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened observations back to 2D grid format.
        
        Args:
            observations: Observations of shape (batch_size, 4096, 3)
            
        Returns:
            Grid tensor of shape (batch_size, 64, 64) with color values
        """
        batch_size = observations.shape[0]
        # Extract color channel (index 2) and reshape to 2D grid
        colors = observations[:, :, 2]  # Shape: (batch_size, 4096)
        grids = colors.view(batch_size, 64, 64)  # Shape: (batch_size, 64, 64)
        return grids
    
    @staticmethod
    def normalize_coordinates(observations: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinate features to [0, 1] range.
        
        Args:
            observations: Observations of shape (batch_size, 4096, 3)
            
        Returns:
            Normalized observations with coordinates in [0, 1]
        """
        normalized = observations.clone().float()
        # Normalize x, y coordinates (first two channels) to [0, 1]
        normalized[:, :, :2] = normalized[:, :, :2] / 63.0
        return normalized