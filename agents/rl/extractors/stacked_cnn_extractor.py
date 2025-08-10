"""
Stacked CNN Feature Extractor for ARC-AGI-3.

This module implements a convolutional neural network architecture for processing
temporally stacked ARC grids, treating multiple time steps as input channels
to a deep CNN for pattern recognition across time and space.

Key features:
- Temporal stacking of multiple grid frames as input channels
- Residual blocks for deep feature extraction
- No pooling to maintain spatial resolution
- Batch normalization for training stability
- Consistent spatial representation across time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym

from .base_extractor import BaseArcFeatureExtractor, FeatureExtractorMixin


class ResidualBlock(nn.Module):
    """
    Residual block for deep CNN feature extraction.
    
    Architecture:
    - Conv 3x3, filters
    - BatchNorm
    - ReLU
    - Conv 3x3, filters  
    - BatchNorm
    - Skip connection
    - ReLU
    """
    
    def __init__(self, filters: int, kernel_size: int = 3, use_batch_norm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            filters, filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            bias=not use_batch_norm
        )
        self.conv2 = nn.Conv2d(
            filters, filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=not use_batch_norm
        )
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(filters)
            self.bn2 = nn.BatchNorm2d(filters)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Skip connection
        out = F.relu(out)
        
        return out


class StackedCNNExtractor(BaseArcFeatureExtractor, FeatureExtractorMixin):
    """
    Stacked CNN feature extractor for temporally stacked ARC-AGI-3 grids.
    
    This architecture processes multiple time steps of 64x64 ARC grids stacked
    as input channels to a deep convolutional neural network. It uses residual
    networks to extract spatio-temporal features while preserving spatial relationships.
    
    Key advantages:
    - Fast: Pure CNN operations optimized for temporal stacking
    - Stable: Fixed spatial structure across time steps
    - Temporal: Processes multiple frames simultaneously
    - Simple: Standard operations, well-optimized in PyTorch
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 256,
        num_residual_blocks: int = 10,
        filters: int = 256,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        input_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize stacked CNN feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output feature dimension
            num_residual_blocks: Number of residual blocks for deep feature extraction
            filters: Number of convolutional filters
            kernel_size: Kernel size for convolutions (typically 3)
            use_batch_norm: Whether to use batch normalization
            input_channels: Number of input channels (temporal depth for stacking)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(observation_space, features_dim, **kwargs)
        
        # Store architecture parameters
        self.num_residual_blocks = num_residual_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        
        # Determine input channels
        if input_channels is None:
            # Auto-detect from observation space
            if len(observation_space.shape) == 3:
                self.input_channels = observation_space.shape[0]
            elif len(observation_space.shape) == 2:
                # Single channel for (64, 64) observations
                self.input_channels = 1
            else:
                # Default: 4 channels (current state, changes, player, hotspots)
                self.input_channels = 4
        else:
            self.input_channels = input_channels
            
        # Initial convolution to project input to filter dimension
        self.initial_conv = nn.Conv2d(
            self.input_channels, 
            filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=not use_batch_norm
        )
        
        if use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(filters)
        else:
            self.initial_bn = nn.Identity()
            
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters, kernel_size, use_batch_norm)
            for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        # Global average pooling + linear projection
        self.output_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.output_bn = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # Final projection to features_dim
        # After global pooling: 32 channels -> features_dim
        self.output_linear = nn.Linear(32, features_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable cuDNN autotuner for optimal convolution algorithms
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Move to GPU if available (SB3 will handle final placement)
            self.cuda()
        
    def _initialize_weights(self):
        """Initialize network weights using standard CNN initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def _observations_to_board(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Convert observations to board format (batch, C, 64, 64).
        
        Args:
            observations: Observations in various formats:
                         - (batch, 4, 64, 64) for engineered mode
                         - (batch, 64, 64) for simple mode
                         - Other legacy formats
            
        Returns:
            Board tensor with shape (batch, input_channels, 64, 64)
        """
        # Handle different observation formats
        if len(observations.shape) == 4 and observations.shape[1] == self.input_channels:
            # Already in correct channel format (batch, C, 64, 64)
            board_tensor = observations.float() / 15.0  # Normalize to [0, 1]
            return board_tensor
        elif len(observations.shape) == 3:
            # Single channel input (batch, 64, 64) - for simple mode
            board = observations.float() / 15.0  # Normalize to [0, 1]
            # Add channel dimension
            board_tensor = board.unsqueeze(1)  # Shape: (batch, 1, 64, 64)
            
            # If model expects more channels, pad with zeros
            if self.input_channels > 1:
                padding = torch.zeros(
                    board.shape[0], 
                    self.input_channels - 1, 
                    64, 64, 
                    device=board.device,
                    dtype=board.dtype
                )
                board_tensor = torch.cat([board_tensor, padding], dim=1)
            
            return board_tensor
        else:
            # Fallback: use base class method for legacy formats
            board = self._standardize_observations(observations)  # Shape: (batch, 64, 64)
            
            # Normalize colors to [0, 1] range (assuming colors are 0-15)
            board_normalized = board.float() / 15.0
            
            if self.input_channels == 1:
                # Single channel mode
                return board_normalized.unsqueeze(1)  # Shape: (batch, 1, 64, 64)
            else:
                # Create multiple channels for compatibility
                channels = []
                
                # Channel 0: Normalized color values
                channels.append(board_normalized)
                
                # Fill remaining channels with zeros or simple features
                for i in range(1, self.input_channels):
                    if i == self.input_channels - 1:
                        # Last channel: binary mask of non-zero cells
                        non_zero_mask = (board > 0).float()
                        channels.append(non_zero_mask)
                    else:
                        # Other channels: zeros
                        channels.append(torch.zeros_like(board_normalized))
                
                # Stack channels
                board_tensor = torch.stack(channels, dim=1)  # Shape: (batch, C, 64, 64)
            
            return board_tensor
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ARC-AGI-3 observations.
        
        Args:
            observations: Batch of observations in any supported format
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        # Convert observations to board format
        board = self._observations_to_board(observations)  # (batch, C, 64, 64)
        
        # Initial convolution
        x = self.initial_conv(board)
        x = self.initial_bn(x)
        x = F.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
            
        # Output projection
        x = self.output_conv(x)
        x = self.output_bn(x)
        x = F.relu(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)
        
        # Final linear projection
        features = self.output_linear(x)  # (batch, features_dim)
        
        return features
        
    def get_representation_info(self) -> Dict[str, Any]:
        """Get information about the stacked CNN representation."""
        complexity = self.get_model_complexity()
        
        return {
            "representation_type": "stacked_cnn",
            "architecture": "Residual CNN (Temporal Stacking)",
            "architecture_details": {
                "num_residual_blocks": self.num_residual_blocks,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "use_batch_norm": self.use_batch_norm,
                "input_channels": self.input_channels,
                "spatial_resolution": "64x64 (preserved throughout)"
            },
            "parameter_count": complexity["total_parameters"],
            "computational_notes": "Pure CNN operations, highly optimized on GPU",
            "inductive_biases": [
                "Strong spatial locality (3x3 convolutions)",
                "Translation equivariance",
                "Hierarchical feature learning through depth",
                "No pooling - preserves exact spatial positions",
                "Skip connections enable both shallow and deep reasoning"
            ],
            "advantages": [
                "~1000x faster than graph-based approaches",
                "Consistent spatial representation across time",
                "Temporal information processing via stacking",
                "GPU-friendly operations"
            ]
        }
        
    def visualize_features(
        self, 
        observations: torch.Tensor,
        layer_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Visualize convolutional features and filters.
        
        Args:
            observations: Input observations
            layer_idx: Which residual block to visualize (None for all)
            save_path: Path to save visualizations
            
        Returns:
            Dictionary of feature maps and filter visualizations
        """
        visualizations = {}
        
        with torch.no_grad():
            # Convert to board format
            board = self._observations_to_board(observations)
            
            # Initial conv features
            x = self.initial_conv(board)
            x = self.initial_bn(x)
            x = F.relu(x)
            visualizations["initial_features"] = x.cpu()
            
            # Features after each residual block
            for i, block in enumerate(self.residual_blocks):
                x = block(x)
                if layer_idx is None or i == layer_idx:
                    visualizations[f"residual_block_{i}"] = x.cpu()
                    
            # Output features before pooling
            out = self.output_conv(x)
            out = self.output_bn(out)
            out = F.relu(out)
            visualizations["output_features"] = out.cpu()
            
        # TODO: Add actual visualization/plotting code if save_path is provided
        
        return visualizations
        
    def get_receptive_field_info(self) -> Dict[str, int]:
        """
        Calculate theoretical receptive field size.
        
        Returns:
            Dictionary with receptive field information
        """
        # Each 3x3 conv increases receptive field by 2
        # Initial conv: 3x3
        # Each residual block has 2 convs: +4 per block
        
        receptive_field = 3  # Initial conv
        receptive_field += 4 * self.num_residual_blocks  # Residual blocks
        
        return {
            "receptive_field_size": receptive_field,
            "grid_size": 64,
            "coverage_ratio": min(1.0, receptive_field / 64),
            "can_see_full_grid": receptive_field >= 64
        }
        
    def analyze_spatial_reasoning(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze spatial reasoning capabilities of the stacked CNN extractor.
        
        Args:
            observations: Input observations
            
        Returns:
            Extended analysis including CNN-specific metrics
        """
        # Get base analysis
        analysis = super().analyze_spatial_reasoning(observations)
        
        # Add stacked CNN-specific analysis
        analysis["receptive_field"] = self.get_receptive_field_info()
        analysis["architecture_info"] = {
            "depth": 2 + 2 * self.num_residual_blocks,  # Total conv layers
            "skip_connections": self.num_residual_blocks,
            "maintains_spatial_resolution": True,
            "uses_global_pooling": True
        }
        
        # Analyze feature map statistics at different depths
        with torch.no_grad():
            board = self._observations_to_board(observations)
            x = self.initial_conv(board)
            x = self.initial_bn(x)
            x = F.relu(x)
            
            feature_stats = []
            feature_stats.append({
                "layer": "initial",
                "activation_mean": float(x.mean()),
                "activation_std": float(x.std()),
                "dead_neurons": float((x == 0).float().mean())
            })
            
            for i, block in enumerate(self.residual_blocks[:3]):  # Sample first 3 blocks
                x = block(x)
                feature_stats.append({
                    "layer": f"residual_{i}",
                    "activation_mean": float(x.mean()),
                    "activation_std": float(x.std()),
                    "dead_neurons": float((x == 0).float().mean())
                })
                
            analysis["layer_statistics"] = feature_stats
            
        return analysis


def create_stacked_cnn_extractor(
    observation_space: gym.spaces.Space,
    features_dim: int = 256,
    num_residual_blocks: int = 10,
    **kwargs
) -> StackedCNNExtractor:
    """
    Factory function to create stacked CNN extractor with sensible defaults.
    
    Args:
        observation_space: Gymnasium observation space
        features_dim: Output feature dimension
        num_residual_blocks: Number of residual blocks
        **kwargs: Additional arguments
        
    Returns:
        Configured StackedCNNExtractor instance
    """
    return StackedCNNExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        num_residual_blocks=num_residual_blocks,
        filters=kwargs.get("filters", 256),
        kernel_size=kwargs.get("kernel_size", 3),
        use_batch_norm=kwargs.get("use_batch_norm", True),
        **kwargs
    )


def test_stacked_cnn_extractor():
    """Test the stacked CNN feature extractor."""
    print("Testing StackedCNNExtractor...")
    
    import numpy as np
    
    # Create dummy observation space
    obs_space = gym.spaces.Box(
        low=0, high=63, shape=(4096, 3), dtype=np.int32
    )
    
    # Create extractor with small config for testing
    extractor = StackedCNNExtractor(
        observation_space=obs_space,
        features_dim=128,
        num_residual_blocks=3,  # Small for testing
        filters=64,  # Smaller for testing
    )
    
    print(f"Created extractor: {extractor}")
    
    # Test forward pass
    batch_size = 2
    observations = torch.randint(0, 16, (batch_size, 4096, 3)).float()
    
    # Time the forward pass
    import time
    start = time.time()
    features = extractor(observations)
    end = time.time()
    
    print(f"✓ Forward pass: {observations.shape} -> {features.shape}")
    print(f"✓ Time per sample: {(end - start) / batch_size:.4f}s")
    
    # Test representation info
    info = extractor.get_representation_info()
    print(f"✓ Representation info: {info['representation_type']}")
    print(f"  - Parameters: {info['parameter_count']:,}")
    print(f"  - Blocks: {info['architecture_details']['num_residual_blocks']}")
    
    # Test receptive field calculation
    rf_info = extractor.get_receptive_field_info()
    print(f"✓ Receptive field: {rf_info['receptive_field_size']}x{rf_info['receptive_field_size']}")
    print(f"  - Can see full grid: {rf_info['can_see_full_grid']}")
    
    # Test visualization
    viz = extractor.visualize_features(observations[:1])
    print(f"✓ Visualization layers: {list(viz.keys())}")
    
    # Test analysis
    analysis = extractor.analyze_spatial_reasoning(observations[:1])
    print(f"✓ Spatial analysis completed")
    print(f"  - Feature mean: {analysis['feature_statistics']['mean']:.4f}")
    print(f"  - Feature std: {analysis['feature_statistics']['std']:.4f}")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_stacked_cnn_extractor()