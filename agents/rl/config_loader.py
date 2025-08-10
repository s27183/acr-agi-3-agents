"""
RL Configuration Loader

Loads configuration from config.json with environment variable overrides.
Environment variables take precedence over config file values.
"""

import json
import os
import torch
import multiprocessing
from typing import Dict, Any
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

class RLConfig:
    """RL Configuration loader from JSON config file."""
    
    def __init__(self, config_file: str = None):
        """Load configuration from config.json or specified file.
        
        Args:
            config_file: Optional path to config file. If None, uses default config.json
        """
        self.config_file = config_file or CONFIG_FILE
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load base configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded RL configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.config_file}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file is missing/invalid."""
        return {
            "training": {
                "max_actions": 200,
                "learning_rate": 0.0003,
                "batch_size": 80,
                "n_epochs": 5,
                "gamma": 0.99,
                "n_steps": 80,
                "num_updates": 50,
                "checkpoint_freq": 10,
                "scorecard_id": "",
                "device": "auto"  # auto/cuda/cpu
            },
            "logging": {
                "tensorboard_enabled": True,
                "log_level": "INFO",
                "grid_visualization_enabled": False
            },
            "paths": {
                "models_dir": "./agents/rl/models",
                "logs_dir": "./agents/rl/logs"
            },
            "representation": {
                "num_residual_blocks": 10,
                "filters": 256,
                "kernel_size": 3,
                "use_batch_norm": True
            }
        }
    
    def _validate_config(self) -> None:
        """Validate configuration parameters and relationships."""
        training = self.config['training']
        
        # Validate and set device configuration
        device_setting = training.get('device', 'auto')
        if device_setting == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_setting in ['cuda', 'cpu']:
            self.device = device_setting
            if device_setting == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
        else:
            logger.warning(f"Invalid device setting '{device_setting}', using auto-detection")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Device configuration: {self.device}")
        
        # Apply device-specific optimizations
        self._apply_device_optimizations()
        
        # Validate scorecard_id is a string
        scorecard_id = training.get('scorecard_id', '')
        if not isinstance(scorecard_id, str):
            raise ValueError(f"scorecard_id must be a string, got {type(scorecard_id)}")
        
        # Skip n_steps validation if using phased training (n_steps will be in phases)
        if not training.get('phased_training', False):
            # Validate n_steps <= max_actions
            if 'n_steps' in training and training['n_steps'] > training['max_actions']:
                raise ValueError(
                    f"n_steps ({training['n_steps']}) must be <= max_actions ({training['max_actions']}). "
                    f"PPO rollout length cannot exceed episode length."
                )
        
        # Calculate optimization step frequency with multiple environments
        n_steps = training.get('n_steps', 128)  # Default fallback
        n_envs = self.n_envs  # Use property to get actual n_envs
        opt_step_freq = n_steps * n_envs  # Multiple environments
        
        # Get checkpoint frequency in optimization steps  
        checkpoint_opt_steps = training.get('checkpoint_freq', 10)
        
        # Validate checkpoint frequency against num_updates
        num_updates = training.get('num_updates', 50)
        if checkpoint_opt_steps > 0:
            if checkpoint_opt_steps < 1:
                logger.warning(f"checkpoint_freq ({checkpoint_opt_steps}) should be >= 1 update")
            elif checkpoint_opt_steps >= num_updates:
                logger.warning(f"checkpoint_freq ({checkpoint_opt_steps}) should be < num_updates ({num_updates}) to save any checkpoints")
                logger.warning(f"Adjusting checkpoint_freq to {max(1, num_updates // 4)} updates")
                # Store the adjusted value back to config for consistent access
                training['checkpoint_freq'] = max(1, num_updates // 4)
        
        # Validate n_envs against CPU cores (more important for CPU mode)
        cpu_cores = multiprocessing.cpu_count()
        if self.device == 'cpu' and n_envs != cpu_cores:
            logger.info(f"CPU mode: Consider using n_envs={cpu_cores} (all CPU cores) instead of {n_envs}")
        elif self.device == 'cuda' and n_envs > 2:
            logger.info(f"GPU mode: Consider using n_envs=2 for optimal GPU utilization (currently {n_envs})")
        
        # Log computed values
        total_timesteps = self.total_timesteps
        total_opt_steps = total_timesteps // opt_step_freq
        
        logger.info(f"Training configuration validated:")
        logger.info(f"  - Optimization every {opt_step_freq} env steps ({n_envs} envs × {n_steps} steps)")
        logger.info(f"  - Total timesteps: {total_timesteps} ({total_opt_steps} optimization steps)")
        logger.info(f"  - Checkpoint every {checkpoint_opt_steps} optimization steps")
        logger.info(f"  - Environment config: n_envs={n_envs}, vec_env_type={self.vec_env_type}")
        logger.info(f"  - Final model saving only (no intermediate evaluation)")
        if training.get('phased_training', False):
            logger.info(f"  - Phased training enabled with {len(self.config.get('training_phases', {}))} phases")
    
    @property
    def n_envs(self) -> int:
        """Number of parallel environments per game."""
        # Get configured value, fallback to auto-detection
        configured_n_envs = self.config['training'].get('n_envs', 'auto')
        
        if configured_n_envs == 'auto' or configured_n_envs is None:
            return self._auto_detect_n_envs()
        else:
            return int(configured_n_envs)
    
    @property 
    def vec_env_type(self) -> str:
        """Vectorized environment type ('subprocess', 'dummy', or 'auto')."""
        return self.config['training'].get('vec_env_type', 'auto')
    
    def _apply_device_optimizations(self) -> None:
        """Apply device-specific configuration optimizations."""
        training = self.config['training']
        cpu_cores = multiprocessing.cpu_count()
        
        if self.device == 'cuda':
            # GPU optimizations
            logger.info("🚀 Applying GPU-optimized settings for CNN training")
            
            # Fewer environments to reduce CPU-GPU transfer overhead
            if 'n_envs' not in training:
                training['n_envs'] = 2
                logger.info(f"  - n_envs: 2 (reduced for GPU efficiency)")
            
            # Larger batch size for better GPU utilization
            if training.get('batch_size', 64) < 128:
                original_batch = training.get('batch_size', 64)
                training['batch_size'] = 256
                logger.info(f"  - batch_size: {original_batch} → 256 (increased for GPU throughput)")
            
            # GPU can handle more updates efficiently due to faster training per update
            current_updates = training.get('num_updates', 50)
            if current_updates == 50:  # Only adjust default, don't override explicit settings
                training['num_updates'] = 100
                logger.info(f"  - num_updates: {current_updates} → 100 (GPU trains faster per update)")
                
        else:
            # CPU optimizations
            logger.info("💻 Applying CPU-optimized settings for CNN training")
            
            # Use all CPU cores for maximum parallelism
            if 'n_envs' not in training:
                training['n_envs'] = cpu_cores
                logger.info(f"  - n_envs: {cpu_cores} (using all CPU cores)")
            elif training['n_envs'] != cpu_cores:
                original_envs = training['n_envs']
                training['n_envs'] = cpu_cores
                logger.info(f"  - n_envs: {original_envs} → {cpu_cores} (adjusted to use all CPU cores)")
    
    def _auto_detect_n_envs(self) -> int:
        """Auto-detect optimal number of environments based on platform."""
        cpu_cores = multiprocessing.cpu_count()
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            # GPU production: Lower n_envs to balance CPU rollouts with GPU model inference
            recommended = min(2, max(1, cpu_cores // 4))
            logger.info(f"GPU detected: recommending n_envs={recommended} (CPU cores: {cpu_cores})")
        else:
            # CPU development: Higher n_envs for maximum parallel API utilization
            recommended = min(4, max(1, cpu_cores // 2))
            logger.info(f"CPU-only detected: recommending n_envs={recommended} (CPU cores: {cpu_cores})")
        
        return recommended
    
    def _auto_detect_vec_env_type(self) -> str:
        """Auto-detect vectorized environment type based on n_envs and device."""
        n_envs = self.n_envs
        
        if n_envs == 1:
            return 'dummy'
        else:
            # Always use subprocess for parallel environments
            return 'subprocess'
    
    @property
    def max_actions(self) -> int:
        """Maximum actions per agent."""
        return self.config['training']['max_actions']
    
    @property
    def learning_rate(self) -> float:
        """PPO learning rate."""
        return self.config['training']['learning_rate']
    
    @property
    def batch_size(self) -> int:
        """PPO batch size."""
        return self.config['training']['batch_size']
    
    @property
    def n_epochs(self) -> int:
        """PPO number of epochs."""
        return self.config['training']['n_epochs']
    
    @property
    def gamma(self) -> float:
        """PPO discount factor."""
        return self.config['training']['gamma']
    
    @property
    def ent_coef(self) -> float:
        """PPO entropy coefficient."""
        return self.config['training'].get('ent_coef', 0.01)
    
    @property
    def vf_coef(self) -> float:
        """PPO value function coefficient."""
        return self.config['training'].get('vf_coef', 0.5)
    
    @property
    def clip_range(self) -> float:
        """PPO clipping range."""
        return self.config['training'].get('clip_range', 0.2)
    
    @property
    def checkpoint_freq(self) -> int:
        """Checkpoint save frequency in updates."""
        return self.config['training'].get('checkpoint_freq', 25)
    
    @property 
    def pytorch_cuda_alloc_conf(self) -> str:
        """PyTorch CUDA memory allocation configuration."""
        return self.config['training'].get('pytorch_cuda_alloc_conf', 'expandable_segments:True')
    
    @property
    def n_steps(self) -> int:
        """PPO number of steps."""
        # For phased training, return a default or phase 1 value
        if self.config['training'].get('phased_training', False):
            phases = self.config.get('training_phases', {})
            phase1 = phases.get('phase1', {})
            return phase1.get('n_steps', 128)  # Default fallback
        return self.config['training'].get('n_steps', 128)
    
    @property
    def num_updates(self) -> int:
        """Number of optimization updates to perform."""
        return self.config['training'].get('num_updates', 50)
    
    @property
    def scorecard_id(self) -> str:
        """Scorecard ID to use for training (empty string = auto-create)."""
        return self.config['training'].get('scorecard_id', '')
    
    @property
    def inference_scorecard_id(self) -> str:
        """Scorecard ID to use for inference (empty string = use swarm-provided scorecard)."""
        return self.config.get('inference', {}).get('scorecard_id', '')
    
    @property
    def inference_config(self) -> Dict[str, Any]:
        """Get inference configuration section."""
        return self.config.get('inference', {})
    
    @property
    def inference_max_actions(self) -> int:
        """Maximum actions for inference (falls back to training max_actions)."""
        return self.inference_config.get('max_actions', self.max_actions)
    
    @property
    def total_timesteps(self) -> int:
        """Total timesteps for training (computed from i)."""
        return self.n_envs*self.n_steps * self.num_updates
    
    
    
    @property
    def tensorboard_enabled(self) -> bool:
        """Whether TensorBoard logging is enabled."""
        return self.config['logging']['tensorboard_enabled']
    
    @property
    def grid_visualization_enabled(self) -> bool:
        """Whether grid visualization logging is enabled."""
        return self.config['logging'].get('grid_visualization_enabled', False)
    
    @property
    def models_dir(self) -> str:
        """Models directory path."""
        return self.config['paths']['models_dir']
    
    @property
    def logs_dir(self) -> str:
        """Logs directory path."""
        return self.config['paths']['logs_dir']
    
    
    @property
    def representation_config(self) -> Dict[str, Any]:
        """Get fixed stacked_10 representation configuration."""
        # Start with user-configurable parameters
        config = self.config.get('representation', {}).copy()
        
        # Fixed architectural parameters for (10, 64, 64) stacked observations
        config.update({
            'policy_type': 'cnn',       # Fixed CNN architecture
            'input_channels': 10,       # Fixed 10-channel input
            'temporal_depth': 10,       # Fixed temporal stack depth
            'description': 'Fixed 10-channel temporal stacking'
        })
        
        return config
    
    @property 
    def policy_config(self) -> Dict[str, Any]:
        """Policy configuration (backward compatibility)."""
        # For backward compatibility, return the active representation config
        return self.representation_config
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get full training configuration."""
        return self.config.get('training', {})
    
    @property
    def dynamic_rollout(self) -> SimpleNamespace:
        """Get dynamic rollout configuration as an object with attributes."""
        config_dict = self.config.get('dynamic_rollout', {
            'enabled': False,
            'target_trajectories_per_update': 4,
            'max_rollout_steps': 1000,
            'movement_threshold_multiplier': 1.0
        })
        return SimpleNamespace(**config_dict)
    
    @property
    def temporal_depth(self) -> int:
        """Get temporal depth for stacked observations (always 10)."""
        return 10
    

# Global config instance
_config_instance = None

def get_config(config_file: str = None) -> RLConfig:
    """Get the global RL configuration instance.
    
    Args:
        config_file: Optional config filename. If None, uses RL_CONFIG env var
                    or defaults to 'config.json'
        
    Returns:
        RLConfig instance
    """
    global _config_instance
    
    # Determine config file to use
    if config_file is None:
        config_file = os.getenv("RL_CONFIG", "config.json")
    
    # If just a filename (not full path), prepend the directory
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), config_file)
    
    if _config_instance is None or config_file != getattr(_config_instance, 'config_file', None):
        _config_instance = RLConfig(config_file)
    return _config_instance