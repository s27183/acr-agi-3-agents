"""
Registry system for managing different feature extractor representations.

This module provides a centralized registry for feature extractors, enabling
easy switching between different neural network architectures via configuration.
"""

import torch
from typing import Dict, Type, Any, Optional, List
from .base_extractor import BaseArcFeatureExtractor
import gymnasium as gym


class RepresentationRegistry:
    """
    Registry for managing feature extractor representations.
    
    This class provides a factory pattern for creating feature extractors,
    allowing easy switching between different architectures via configuration.
    
    Features:
    - Dynamic registration of extractor classes
    - Configuration-based instantiation
    - Type validation and error handling
    - Support for custom hyperparameters per representation type
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._extractors: Dict[str, Type[BaseArcFeatureExtractor]] = {}
        self._default_configs: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self, 
        name: str, 
        extractor_class: Type[BaseArcFeatureExtractor],
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a feature extractor class.
        
        Args:
            name: Identifier for the extractor (e.g., "gps_sage", "cnn", "vit")
            extractor_class: The extractor class to register
            default_config: Default configuration parameters for this extractor
            
        Raises:
            ValueError: If name is already registered or class is invalid
        """
        # Validate name
        if not isinstance(name, str) or not name:
            raise ValueError("Extractor name must be a non-empty string")
        
        if name in self._extractors:
            raise ValueError(f"Extractor '{name}' is already registered")
        
        # Validate class
        if not issubclass(extractor_class, BaseArcFeatureExtractor):
            raise ValueError(
                f"Extractor class must inherit from BaseArcFeatureExtractor, "
                f"got {extractor_class}"
            )
        
        # Register extractor
        self._extractors[name] = extractor_class
        self._default_configs[name] = default_config or {}
        
        print(f"✅ Registered extractor: {name} -> {extractor_class.__name__}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a feature extractor.
        
        Args:
            name: Name of the extractor to unregister
        """
        if name in self._extractors:
            del self._extractors[name]
            del self._default_configs[name]
            print(f"🗑️ Unregistered extractor: {name}")
        else:
            print(f"⚠️ Extractor '{name}' not found in registry")
    
    def create_extractor(
        self,
        name: str,
        observation_space: gym.spaces.Space,
        features_dim: int = 256,
        **kwargs
    ) -> BaseArcFeatureExtractor:
        """
        Create a feature extractor instance.
        
        Args:
            name: Name of the registered extractor
            observation_space: Gymnasium observation space
            features_dim: Output feature dimension
            **kwargs: Additional parameters for the extractor
            
        Returns:
            Initialized feature extractor instance
            
        Raises:
            ValueError: If extractor name is not registered
            Exception: If extractor creation fails
        """
        if name not in self._extractors:
            available = list(self._extractors.keys())
            raise ValueError(
                f"Unknown extractor '{name}'. Available extractors: {available}"
            )
        
        # Get extractor class and default config
        extractor_class = self._extractors[name]
        default_config = self._default_configs[name].copy()
        
        # Merge default config with provided kwargs
        config = default_config
        config.update(kwargs)
        
        try:
            # Create extractor instance
            extractor = extractor_class(
                observation_space=observation_space,
                features_dim=features_dim,
                **config
            )
            
            print(f"✅ Created {name} extractor: {extractor}")
            return extractor
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {name} extractor: {e}\n"
                f"Class: {extractor_class}\n"
                f"Config: {config}"
            ) from e
    
    def get_available_extractors(self) -> List[str]:
        """
        Get list of available extractor names.
        
        Returns:
            List of registered extractor names
        """
        return list(self._extractors.keys())
    
    def get_extractor_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered extractor.
        
        Args:
            name: Name of the extractor
            
        Returns:
            Dictionary with extractor information
            
        Raises:
            ValueError: If extractor name is not registered
        """
        if name not in self._extractors:
            raise ValueError(f"Unknown extractor '{name}'")
        
        extractor_class = self._extractors[name]
        default_config = self._default_configs[name]
        
        return {
            "name": name,
            "class": extractor_class.__name__,
            "module": extractor_class.__module__,
            "default_config": default_config,
            "docstring": extractor_class.__doc__,
        }
    
    def is_registered(self, name: str) -> bool:
        """
        Check if an extractor is registered.
        
        Args:
            name: Name to check
            
        Returns:
            True if extractor is registered
        """
        return name in self._extractors
    
    def list_extractors(self) -> None:
        """Print information about all registered extractors."""
        if not self._extractors:
            print("No extractors registered")
            return
        
        print("📋 Registered Feature Extractors:")
        print("=" * 50)
        
        for name in sorted(self._extractors.keys()):
            info = self.get_extractor_info(name)
            print(f"🔧 {name}")
            print(f"   Class: {info['class']}")
            print(f"   Module: {info['module']}")
            if info['default_config']:
                print(f"   Default Config: {info['default_config']}")
            print()
    
    def validate_config(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration for an extractor.
        
        Args:
            name: Name of the extractor
            config: Configuration dictionary to validate
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if name not in self._extractors:
            raise ValueError(f"Unknown extractor '{name}'")
        
        # Start with default config
        validated_config = self._default_configs[name].copy()
        
        # Update with provided config
        validated_config.update(config)
        
        # Add representation-specific validation here if needed
        # For now, just return the merged config
        return validated_config
    
    def create_from_config(
        self,
        config: Dict[str, Any],
        observation_space: gym.spaces.Space,
        features_dim: int = 256
    ) -> BaseArcFeatureExtractor:
        """
        Create extractor from configuration dictionary.
        
        Args:
            config: Configuration with 'representation_type' and other parameters
            observation_space: Gymnasium observation space
            features_dim: Output feature dimension
            
        Returns:
            Initialized feature extractor
            
        Raises:
            ValueError: If configuration is missing required fields
        """
        if 'representation_type' not in config:
            raise ValueError("Config must contain 'representation_type' field")
        
        representation_type = config['representation_type']
        
        # Extract representation-specific config
        extractor_config = {k: v for k, v in config.items() if k != 'representation_type'}
        
        return self.create_extractor(
            name=representation_type,
            observation_space=observation_space,
            features_dim=features_dim,
            **extractor_config
        )
    
    def __len__(self) -> int:
        """Return number of registered extractors."""
        return len(self._extractors)
    
    def __contains__(self, name: str) -> bool:
        """Check if extractor name is registered."""
        return name in self._extractors
    
    def __iter__(self):
        """Iterate over registered extractor names."""
        return iter(self._extractors.keys())
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        count = len(self._extractors)
        extractors = list(self._extractors.keys())
        return f"RepresentationRegistry({count} extractors: {extractors})"


# Default registry instance
_default_registry = RepresentationRegistry()


def get_default_registry() -> RepresentationRegistry:
    """
    Get the default global registry instance.
    
    Returns:
        The default RepresentationRegistry instance
    """
    return _default_registry


def register_extractor(
    name: str,
    extractor_class: Type[BaseArcFeatureExtractor],
    default_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register an extractor in the default registry.
    
    Args:
        name: Identifier for the extractor
        extractor_class: The extractor class to register
        default_config: Default configuration parameters
    """
    _default_registry.register(name, extractor_class, default_config)


def create_extractor(
    name: str,
    observation_space: gym.spaces.Space,
    features_dim: int = 256,
    **kwargs
) -> BaseArcFeatureExtractor:
    """
    Create an extractor using the default registry.
    
    Args:
        name: Name of the registered extractor
        observation_space: Gymnasium observation space
        features_dim: Output feature dimension
        **kwargs: Additional parameters
        
    Returns:
        Initialized feature extractor
    """
    return _default_registry.create_extractor(
        name=name,
        observation_space=observation_space,
        features_dim=features_dim,
        **kwargs
    )


def list_available_extractors() -> List[str]:
    """
    Get list of available extractors from default registry.
    
    Returns:
        List of registered extractor names
    """
    return _default_registry.get_available_extractors()


def test_registry():
    """Test function for the representation registry."""
    print("Testing RepresentationRegistry...")
    
    # Create test registry
    registry = RepresentationRegistry()
    
    # Test empty registry
    assert len(registry) == 0
    assert registry.get_available_extractors() == []
    
    # Create dummy extractor for testing
    class DummyExtractor(BaseArcFeatureExtractor):
        def forward(self, observations):
            return torch.randn(observations.shape[0], self.features_dim)
        
        def get_representation_info(self):
            return {"representation_type": "dummy", "test": True}
    
    # Test registration
    registry.register("dummy", DummyExtractor, {"param1": 42})
    assert len(registry) == 1
    assert "dummy" in registry
    assert registry.is_registered("dummy")
    
    # Test extractor creation
    import gymnasium as gym
    obs_space = gym.spaces.Box(low=0, high=63, shape=(4096, 3), dtype=torch.int32)
    
    try:
        extractor = registry.create_extractor("dummy", obs_space, features_dim=128)
        assert extractor.features_dim == 128
        print("✅ Registry test passed")
        return True
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False


if __name__ == "__main__":
    test_registry()