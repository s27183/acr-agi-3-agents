"""
RLTrainingSwarmGymnasium - Gymnasium-based RL training with swarm orchestration

This module provides RL training through the main.py swarm system, ensuring proper
API connectivity, scorecard management, and cleanup while using our modern 
multi-representation architecture.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from ..swarm import Swarm
from .config_loader import get_config
from .dynamic_ppo_trainer import DynamicPPOTrainer

logger = logging.getLogger(__name__)


class RLTrainingSwarmGymnasium(Swarm):
    """
    Gymnasium-based RL training swarm that integrates with main.py orchestration.
    
    This class extends the base Swarm to provide RL training capabilities while
    maintaining all the benefits of swarm orchestration (API connectivity, 
    scorecard management, cleanup, etc.).
    
    Features:
    - Multi-representation support (CNN with temporal stacking)
    - Config-driven representation selection
    - Proper vectorized environment support
    - Integration with main.py workflow
    - Full swarm orchestration benefits
    """
    
    def __init__(self, agent: str, ROOT_URL: str, games: List[str], tags: Optional[List[str]] = None, config_file: Optional[str] = None):
        """
        Initialize RL training swarm.
        
        Args:
            agent: Agent type ('rltraining' only - standard PPO)
            ROOT_URL: API root URL
            games: List of game IDs to train on
            tags: Optional list of tags for scorecard
            config_file: Optional config file to use (defaults to config.json)
        """
        super().__init__(agent, ROOT_URL, games, tags=tags)
        
        # Load RL configuration
        self.config_file = config_file
        self.config = get_config(config_file)
        self.representation_type = "stacked_10"  # Fixed representation
        
        logger.info(f"RLTrainingSwarmGymnasium initialized:")
        logger.info(f"  - Representation: {self.representation_type} (fixed)")
        logger.info(f"  - Games: {games}")
        logger.info(f"  - Tags: {tags}")
        logger.info(f"  - Training config: {self.config.n_envs} envs, {self.config.total_timesteps} timesteps")
        
        # Training state
        self.training_complete = False
        self.model = None
        
    def _create_vectorized_env(self, make_env, vec_env_type, n_envs):
        """
        Create vectorized environment with fallback support.
        
        Args:
            make_env: environmnet factory function
            vec_env_type: Type of vectorized environment ('subprocess', 'dummy')
            n_envs: Number of environments
            
        Returns:
            Vectorized environment instance
        """
        
        if vec_env_type == 'subprocess' and n_envs > 1:
            try:
                logger.info(f"Creating SubprocVecEnv with {n_envs} environments...")
                env = SubprocVecEnv([
                        make_env(env_id=i, rank=i) for i in range(n_envs)
                ])
                logger.info(f"✅ SubprocVecEnv created successfully")
                return env
            except Exception as e:
                logger.warning(f"SubprocVecEnv creation failed: {e}")
                logger.warning("Falling back to DummyVecEnv")
        
        # Fallback to DummyVecEnv
        logger.info(f"Creating DummyVecEnv with {n_envs} environment(s)...")
        env = DummyVecEnv(
                [
                        make_env(env_id=i, rank=i) for i in range(n_envs)
                ]
        )
        logger.info(f"✅ DummyVecEnv created successfully")
        return env
        
    def main(self):
        """
        Main training loop using swarm orchestration.
        
        This method runs the complete RL training process while maintaining
        all swarm benefits like scorecard management and proper cleanup.
        """
        if len(self.GAMES) != 1:
            logger.error("RL training currently supports exactly one game at a time")
            logger.error(f"Requested games: {self.GAMES}")
            raise ValueError("RL training requires exactly one game")
        
        game_id = self.GAMES[0]
        logger.info(f"Starting RL training for game: {game_id}")
        
        # Handle scorecard creation - use config scorecard_id if provided, otherwise create new
        if self.config.scorecard_id and self.config.scorecard_id.strip():
            # Use provided scorecard ID from config
            self.card_id = self.config.scorecard_id.strip()
            logger.info(f"Using configured scorecard ID: {self.card_id}")
        else:
            # Create new scorecard as before
            self.card_id = self.open_scorecard()
            logger.info(f"Created new scorecard: {self.card_id}")
        
        # Set up game-specific logging
        clean_game_name = game_id.split('-')[0]
        
        # Ensure logs directory exists
        os.makedirs(self.config.logs_dir, exist_ok=True)
        log_file_path = os.path.join(self.config.logs_dir, f"training_{clean_game_name}.log")
        
        # Add file handler for training logs
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Generate scorecard URL for easy access (now logged to both console and file)
        if self.card_id and '.' in self.card_id:
            card_id_for_url = self.card_id.replace('.', '-')
            scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id_for_url}"
            logger.info(f"Scorecard URL: {scorecard_url}")
        else:
            logger.info(f"Scorecard URL: {self.ROOT_URL}/scorecards/{self.card_id}")
        
        # Configure Stable Baselines3 logging to use the same file
        # This captures PPO training metrics (FPS, loss, etc.) in our log file
        sb3_logger = logging.getLogger("stable_baselines3")
        sb3_logger.setLevel(logging.INFO)
        sb3_logger.addHandler(file_handler)
        
        # Also capture warnings and other relevant loggers including our RL training modules
        additional_loggers = [
            "stable_baselines3.common.callbacks", 
            "stable_baselines3.ppo", 
            "gymnasium",
            "agents.rl.dynamic_ppo_trainer",
            "agents.rl.dynamic_trajectory_buffer"
        ]
        for logger_name in additional_loggers:
            log_obj = logging.getLogger(logger_name)
            log_obj.addHandler(file_handler)
        
        logger.info(f"Unified training log file: {log_file_path}")
        logger.info(f"Python logging (app + SB3) will be captured in: {log_file_path}")
        
        try:
            # Import training dependencies (may not be available in all environments)
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.callbacks import CheckpointCallback
            from stable_baselines3.common.logger import configure
            from .arc_grid_env import ARCGridEnvironment
            # Policy configuration helper (inline replacement for removed GPS+SAGE module)
            
        except ImportError as e:
            logger.error(f"RL training dependencies not available: {e}")
            logger.error("Install with: uv sync --active")
            return

        
        logger.info(f"Training configuration:")
        logger.info(f"  - Algorithm: PPO (fixed)")
        logger.info(f"  - Fixed observation: (10, 64, 64) stacked temporal grids")
        logger.info(f"  - Representation: {self.representation_type}")

        # Create environment factory for vectorized training
        def make_env(env_id: int = 0, rank:int=0, seed: int = 0):
            def _init():
                # Configure multiple loggers that operate in subprocess environments FIRST
                loggers_to_configure = [
                    'agents.rl.arc_grid_env',
                    'agents.rl.dynamic_trajectory_buffer'
                ]

                for logger_name in loggers_to_configure:
                    env_logger = logging.getLogger(logger_name)
                    env_logger.setLevel(logging.INFO)

                    # Add file handler if not already present
                    log_file_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'training_{clean_game_name}.log')
                                          for h in env_logger.handlers)
                    if not log_file_exists:
                        env_file_handler = logging.FileHandler(log_file_path, mode='a')
                        env_file_handler.setLevel(logging.INFO)
                        env_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
                        env_file_handler.setFormatter(env_formatter)
                        env_logger.addHandler(env_file_handler)

                # Also configure the calibration logger specifically
                calibration_logger = logging.getLogger('agents.rl.calibration')
                calibration_logger.setLevel(logging.INFO)
                cal_log_file_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'training_{clean_game_name}.log')
                                      for h in calibration_logger.handlers)
                if not cal_log_file_exists:
                    cal_file_handler = logging.FileHandler(log_file_path, mode='a')
                    cal_file_handler.setLevel(logging.INFO)
                    cal_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
                    cal_file_handler.setFormatter(cal_formatter)
                    calibration_logger.addHandler(cal_file_handler)

                env = ARCGridEnvironment(
                    game_id=game_id,
                    max_steps=self.config.max_actions,
                    tags=self.tags,
                    env_id=env_id,
                    scorecard_id=self.card_id,
                    root_url=self.ROOT_URL,
                    api_key=os.getenv("ARC_API_KEY"),
                    config_file=self.config_file  # Pass config file for subprocess inheritance
                )
                # Now reset with logging properly configured
                env.reset(seed+rank)
                return env
            set_random_seed(seed)
            return _init
        
        # Create vectorized environment based on config
        n_envs = self.config.n_envs
        vec_env_type = self.config.vec_env_type
        
        # Auto-detect vec_env_type if needed
        if vec_env_type == 'auto':
            vec_env_type = self.config._auto_detect_vec_env_type()
        
        # Create vectorized environment with fallback
        env = self._create_vectorized_env(make_env, vec_env_type, n_envs)
        
        logger.info(f"Created vectorized environment: {vec_env_type} with {n_envs} environments")

        
        # Standard PPO uses our custom feature extractors
        policy_config = create_policy_config_from_representation(
            representation_type=self.representation_type,
            features_dim=256,
            **self.config.representation_config
        )
        
        logger.info(f"Using policy configuration: {policy_config}")
        
        # Set up model paths using clean game name
        clean_game_name = game_id.split('-')[0]  # Extract clean game name (ls20 from ls20-f340c8e5...)
        model_dir = os.path.join(self.config.models_dir, clean_game_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Fixed representation for model naming
        active_representation = "stacked_10"  # Fixed representation
        
        # Check for existing models to continue training
        model_path = None
        
        # Priority 1: Current final model
        if os.path.exists(os.path.join(model_dir, "final_model.zip")):
            model_path = os.path.join(model_dir, "final_model.zip")
            logger.info(f"Found existing final model to continue training: {model_path}")
        
        # Priority 2: Legacy models with specific representations (backward compatibility)
        elif os.path.exists(os.path.join(model_dir, "final_model_ppo_stacked_10.zip")):
            model_path = os.path.join(model_dir, "final_model_ppo_stacked_10.zip")
            logger.info(f"Found existing stacked_10 model to continue training: {model_path}")
            logger.warning(f"Loading legacy stacked_10 model. Will save as final_model.zip going forward.")
        elif os.path.exists(os.path.join(model_dir, "final_model_ppo_stacked_8.zip")):
            model_path = os.path.join(model_dir, "final_model_ppo_stacked_8.zip")
            logger.info(f"Found existing stacked_8 model to continue training: {model_path}")
            logger.warning(f"Loading legacy stacked_8 model. Will save as final_model.zip going forward.")
        elif os.path.exists(os.path.join(model_dir, "final_model_ppo_cnn.zip")):
            model_path = os.path.join(model_dir, "final_model_ppo_cnn.zip")
            logger.info(f"Found existing CNN model to continue training: {model_path}")
            logger.warning(f"Loading legacy CNN model. Will save as final_model.zip going forward.")
        
        # Extract policy class and kwargs from config
        policy_class = policy_config.get('policy_class', 'MlpPolicy')
        policy_kwargs = policy_config.get('policy_kwargs', {})
        
        # Detect and configure device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ PPO will use device: {device}")
        if device == 'cuda':
            # Configure CUDA memory management
            if hasattr(self.config, 'pytorch_cuda_alloc_conf'):
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config.pytorch_cuda_alloc_conf
                logger.info(f"🔧 CUDA memory config: {self.config.pytorch_cuda_alloc_conf}")
            
            # Clear any existing GPU memory cache
            torch.cuda.empty_cache()
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Log initial memory usage
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"   Initial GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Create standard PPO model with safe loading
        if model_path:
            logger.info(f"🔄 Loading pretrained PPO model from: {model_path}")
            
            # Standard model loading
            try:
                self.model = PPO.load(
                    model_path, 
                    env=env, 
                    verbose=1,
                    device=device,  # Explicitly set device
                    tensorboard_log=os.path.join(self.config.logs_dir, clean_game_name) if self.config.tensorboard_enabled else None
                )
                logger.info(f"✅ Successfully loaded pretrained PPO model on {device}")
            except Exception as e:
                logger.error(f"❌ Model loading failed: {str(e)}")
                logger.info("🔄 Creating fresh model instead")
                # Create fresh model immediately
                self.model = PPO(
                    policy=policy_class,
                    env=env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=self.config.learning_rate,
                    n_steps=self.config.n_steps,
                    batch_size=self.config.batch_size,
                    n_epochs=self.config.n_epochs,
                    gamma=self.config.gamma,
                    ent_coef=self.config.ent_coef,
                    vf_coef=self.config.vf_coef,
                    clip_range=self.config.clip_range,
                    device=device,  # Explicitly set device
                    verbose=1,  # Ensures training metrics are logged (FPS, loss, etc.)
                    tensorboard_log=os.path.join(self.config.logs_dir, clean_game_name) if self.config.tensorboard_enabled else None
                )
        else:
            logger.info("No pretrained model found, creating fresh PPO model")
            self.model = PPO(
                policy=policy_class,
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                clip_range=self.config.clip_range,
                device=device,  # Explicitly set device
                verbose=1,  # Ensures training metrics are logged (FPS, loss, etc.)
                tensorboard_log=os.path.join(self.config.logs_dir, clean_game_name) if self.config.tensorboard_enabled else None
            )
        
        # Configure SB3 internal logger to write training metrics to files
        # This creates structured data files (CSV, JSON) alongside our unified Python log
        if self.config.tensorboard_enabled:
            # Configure logger for tensorboard, CSV, and JSON data files (no stdout to avoid NULL bytes)
            metrics_path = os.path.join(self.config.logs_dir, clean_game_name)
            os.makedirs(metrics_path, exist_ok=True)
            new_logger = configure(metrics_path, ["tensorboard", "csv", "json"])
        else:
            # Configure logger for CSV and JSON data files only (no stdout to avoid NULL bytes)
            metrics_path = os.path.join(self.config.logs_dir, f"training_data_{clean_game_name}")
            os.makedirs(metrics_path, exist_ok=True)
            new_logger = configure(metrics_path, ["csv", "json"])
        
        # Set the model to use our configured logger
        self.model.set_logger(new_logger)
        
        # Note: Training metrics are now handled by the DynamicPPOTrainer's _write_metrics_table_to_log method
        # This provides clean, formatted metrics tables directly in the log file without NULL bytes
        
        logger.info(f"Created PPO model with representation: {self.representation_type}")
        logger.info(f"Configured comprehensive logging:")
        logger.info(f"  • Python logs (app + SB3): {log_file_path}")
        logger.info(f"  • Training metrics (CSV/JSON): {metrics_path}")
        if self.config.tensorboard_enabled:
            logger.info(f"  • TensorBoard logs: {metrics_path}")
        
        # No callbacks needed - using final model saving only
        logger.info("Using final model saving only (no intermediate evaluation)")
        
        # Start training
        logger.info(f"Starting training for {self.config.total_timesteps} timesteps...")
        logger.info(f"Representation: {self.representation_type}")
        logger.info(f"Scorecard ID: {self.card_id}")
        
        start_time = time.time()
        
        try:
            # Check if dynamic rollouts are enabled
            use_dynamic_rollouts = self.config.training.get('use_dynamic_rollouts', False)
            dynamic_config = self.config.dynamic_rollout
            
            # Standard PPO supports dynamic rollouts
            if use_dynamic_rollouts and dynamic_config and dynamic_config.enabled:
                # Use dynamic PPO trainer
                logger.info("Using Dynamic PPO Trainer with quality-based rollouts")
                logger.info(f"  - Target trajectories per update: {dynamic_config.target_trajectories_per_update}")
                
                dynamic_trainer = DynamicPPOTrainer(
                    model=self.model,
                    env=env,
                    target_trajectories_per_update=dynamic_config.target_trajectories_per_update,
                    max_rollout_steps=dynamic_config.max_rollout_steps
                )
                
                # Set up checkpoint callback for automatic saving
                def checkpoint_callback(update_num):
                    if update_num % self.config.checkpoint_freq == 0:
                        final_model_path = os.path.join(model_dir, "final_model.zip")
                        try:
                            self.model.save(final_model_path.replace('.zip', ''))  # SB3 adds .zip automatically
                            logger.info(f"💾 Checkpoint saved: {final_model_path} (Update {update_num}/{self.config.num_updates})")
                        except Exception as e:
                            logger.warning(f"⚠️ Checkpoint save failed: {e}")
                
                dynamic_trainer.checkpoint_callback = checkpoint_callback
                
                dynamic_trainer.learn(
                    total_timesteps=self.config.total_timesteps,
                    tb_log_name="dynamic_training"
                )
                
                # Log dynamic training statistics
                stats = dynamic_trainer.get_training_stats()
                logger.info(f"Dynamic training stats: avg_quality={stats['avg_quality']:.3f}, "
                           f"updates={stats['update_count']}")
                
            else:
                # Use standard PPO training with fixed rollouts
                logger.info("✅ STANDARD PPO: Using fixed rollouts (dynamic rollouts disabled)")
                
                # Create checkpoint callback for standard PPO
                from stable_baselines3.common.callbacks import BaseCallback
                
                class CheckpointCallback(BaseCallback):
                    def __init__(self, save_freq: int, save_path: str):
                        super().__init__()
                        self.save_freq = save_freq
                        self.save_path = save_path
                        self.n_calls = 0
                    
                    def _on_step(self) -> bool:
                        self.n_calls += 1
                        # Save every save_freq timesteps, converted from update frequency
                        update_num = self.n_calls // (self.training_env.num_envs * self.model.n_steps)
                        if update_num > 0 and update_num % (self.save_freq // (self.training_env.num_envs * self.model.n_steps)) == 0:
                            final_model_path = os.path.join(self.save_path, "final_model")
                            self.model.save(final_model_path)
                            logger.info(f"💾 Checkpoint saved: {final_model_path}.zip (Update {update_num})")
                        return True
                
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.config.checkpoint_freq * self.config.n_steps * self.config.n_envs,
                    save_path=model_dir
                )
                
                self.model.learn(
                    total_timesteps=self.config.total_timesteps,
                    tb_log_name="training",
                    callback=checkpoint_callback
                )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(model_dir, "final_model.zip")
            
            try:
                self.model.save(final_model_path.replace('.zip', ''))  # SB3 adds .zip automatically
                logger.info(f"✅ Final model saved to: {final_model_path}")
                
                # Log training statistics
                training_stats = {
                    'training_time_seconds': training_time,
                    'total_timesteps': self.config.total_timesteps,
                    'representation': active_representation,
                    'completed_successfully': True
                }
                
                # Add dynamic training stats if available
                if use_dynamic_rollouts and 'dynamic_trainer' in locals():
                    training_stats.update(dynamic_trainer.get_training_stats())
                    
                logger.info(f"📊 Training statistics: {training_stats}")
                
            except Exception as e:
                logger.error(f"❌ Model saving failed: {str(e)}")
            
            self.training_complete = True
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save model on interruption
            interrupt_model_path = os.path.join(model_dir, "interrupted_model")
            if self.model:
                self.model.save(interrupt_model_path)  # SB3 adds .zip automatically
                logger.info(f"Model saved to: {interrupt_model_path}.zip")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Clean up environments
            env.close()
            logger.info("Training session cleanup completed")
        
        # Training finished - this will trigger cleanup in main.py
        logger.info("RL training completed, initiating shutdown...")
    
    def cleanup(self, scorecard=None):
        """
        Clean up training resources and provide final report.
        
        Args:
            scorecard: Final scorecard data
        """
        logger.info("=== RL TRAINING SESSION COMPLETE ===")
        
        if scorecard:
            logger.info(f"Final scorecard: {scorecard.model_dump()}")
        
        if self.training_complete:
            logger.info(f"✅ Training completed successfully")
            logger.info(f"✅ Representation: {self.representation_type}")
            logger.info(f"✅ Total timesteps: {self.config.total_timesteps}")
            logger.info(f"✅ Model saved in: {self.config.models_dir}")
        else:
            logger.warning("⚠️  Training did not complete normally")
        
        logger.info("=== SESSION END ===")
        
        # Call parent cleanup
        super().cleanup(scorecard)

# Create policy configuration based on active representation
def create_policy_config_from_representation(representation_type: str, features_dim: int = 256, **kwargs):
    """Create policy configuration for the specified representation type."""
    from .extractors import representation_registry

    if representation_registry.is_registered(representation_type):
        # Use custom feature extractor from registry
        extractor_class = representation_registry._extractors[representation_type]
        config = representation_registry._default_configs[representation_type].copy()
        config.update(kwargs)  # Override with provided kwargs

        return {
            'policy_class': 'MultiInputPolicy',
            'policy_kwargs': {
                'features_extractor_class': extractor_class,
                'features_extractor_kwargs': config,
                'net_arch': [256, 256],  # Standard architecture
            }
        }
    else:
        # Fallback to MlpPolicy for unrecognized representation types
        return {
            'policy_class': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [256, 256]
            }
        }