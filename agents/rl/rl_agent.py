"""
RLAgent - Inference agent for trained RL models.

Loads trained PPO models and plays ARC-AGI-3 games using the swarm infrastructure.
"""

import logging
import os
from typing import Any

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


class RLAgent(Agent):
    """An agent that uses a trained PPO model for inference."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        # Load RL config to check for scorecard override
        self._load_rl_config()
        
        # Override scorecard if specified in config  
        self._override_scorecard_if_configured()
        
        # Override MAX_ACTIONS if specified in inference config
        self._override_max_actions_if_configured()
        
        # Load deterministic setting for inference predictions
        self._load_deterministic_setting()
        
        # Set up dedicated inference logging
        self._setup_inference_logging()
        
        # Initialize RL components
        self.model = None
        self.env = None
        self.current_obs = None
        self.env_needs_reset = True
        
        # Load the trained model for this game
        self._load_trained_model()
        
        # Initialize environment for proper RL inference
        self._setup_inference_env()

    def _load_trained_model(self) -> None:
        """Load the trained PPO model for this game."""
        try:
            from stable_baselines3 import PPO
            
            # Extract clean game name (e.g., ls20 from ls20-f340c8e5...)
            clean_game_name = self.game_id.split('-')[0]
            models_dir = f"./agents/rl/models/{clean_game_name}"
            
            # Try to find a trained model with priority order
            model_path = None
            
            # Priority 1: Current final model
            if os.path.exists(os.path.join(models_dir, "final_model.zip")):
                model_path = os.path.join(models_dir, "final_model.zip")
                logger.info(f"Found final model: {model_path}")
            
            # Priority 2: Legacy models (backward compatibility)
            elif os.path.exists(os.path.join(models_dir, "final_model_ppo_stacked_10.zip")):
                model_path = os.path.join(models_dir, "final_model_ppo_stacked_10.zip")
                logger.info(f"Found legacy stacked_10 model: {model_path}")
            elif os.path.exists(os.path.join(models_dir, "final_model_ppo_stacked_8.zip")):
                model_path = os.path.join(models_dir, "final_model_ppo_stacked_8.zip") 
                logger.info(f"Found legacy stacked_8 model: {model_path}")
            elif os.path.exists(os.path.join(models_dir, "final_model_ppo_stacked_3.zip")):
                model_path = os.path.join(models_dir, "final_model_ppo_stacked_3.zip")
                logger.info(f"Found legacy stacked_3 model: {model_path}")
            elif os.path.exists(os.path.join(models_dir, "final_model_ppo_stacked.zip")):
                model_path = os.path.join(models_dir, "final_model_ppo_stacked.zip")
                logger.info(f"Found legacy stacked model: {model_path}")
            
            if model_path:
                logger.info(f"Loading trained model from: {model_path}")
                self.model = PPO.load(model_path)
                logger.info(f"✅ Successfully loaded PPO model for {clean_game_name}")
            else:
                logger.error(f"❌ No trained model found in: {models_dir}")
                logger.error(f"Train a model first with: uv run main.py --agent=rltraining --game={clean_game_name}")
                raise FileNotFoundError(f"No trained model found for game {clean_game_name}")
                
        except ImportError as e:
            logger.error(f"❌ RL dependencies not available: {e}")
            logger.error("Install with: uv sync --agentops")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _setup_inference_env(self) -> None:
        """Set up environment for proper RL inference."""
        try:
            from .arc_grid_env import ARCGridEnvironment
            
            # Create environment using agent's scorecard to avoid conflicts
            # This allows proper RL inference with env.step() pattern
            self.env = ARCGridEnvironment(
                game_id=self.game_id,
                max_steps=self.MAX_ACTIONS,
                tags=self.tags,
                env_id=0,
                scorecard_id=self.card_id,  # Share agent's scorecard for single game session
                root_url=self.ROOT_URL,
                api_key=os.getenv("ARC_API_KEY")
            )
            logger.info("✅ RL inference environment set up successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to set up RL environment: {e}")
            raise

    def _load_rl_config(self) -> None:
        """Load RL configuration to check for scorecard overrides."""
        try:
            from .config_loader import get_config
            self.config = get_config()
            logger.debug("✅ Successfully loaded RL configuration for inference")
        except Exception as e:
            logger.warning(f"Could not load RL config: {e}, using swarm scorecard")
            self.config = None

    def _override_scorecard_if_configured(self) -> None:
        """Override scorecard ID if inference scorecard is specified in config."""
        if (self.config and 
            hasattr(self.config, 'inference_scorecard_id') and
            self.config.inference_scorecard_id and 
            self.config.inference_scorecard_id.strip()):
            
            original_card_id = self.card_id
            self.card_id = self.config.inference_scorecard_id.strip()
            logger.info(f"🎯 Using configured inference scorecard ID: {self.card_id} (was: {original_card_id})")
        else:
            logger.info(f"📊 Using swarm-provided scorecard ID for inference: {self.card_id}")

    def _override_max_actions_if_configured(self) -> None:
        """Override MAX_ACTIONS if specified in inference config."""
        if (self.config and 
            hasattr(self.config, 'inference_max_actions')):
            
            original_max_actions = self.MAX_ACTIONS
            self.MAX_ACTIONS = self.config.inference_max_actions
            
            if self.MAX_ACTIONS != original_max_actions:
                logger.info(f"🎯 Using inference MAX_ACTIONS: {self.MAX_ACTIONS} (was: {original_max_actions})")
            else:
                logger.info(f"📊 Using default MAX_ACTIONS for inference: {self.MAX_ACTIONS}")

    def _load_deterministic_setting(self) -> None:
        """Load deterministic setting from inference config."""
        if (self.config and 
            hasattr(self.config, 'inference_deterministic')):
            self.deterministic = self.config.inference_deterministic
            logger.info(f"🎯 Using inference deterministic: {self.deterministic}")
        else:
            self.deterministic = True  # Default to deterministic for consistent inference
            logger.info(f"📊 Using default deterministic: {self.deterministic}")

    def _setup_inference_logging(self) -> None:
        """Set up dedicated inference logging for this game."""
        try:
            # Extract clean game name (e.g., ls20 from ls20-f340c8e5...)
            clean_game_name = self.game_id.split('-')[0]
            
            # Use same logs directory as training
            logs_dir = self.config.logs_dir if self.config else "./agents/rl/logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create inference-specific log file
            log_file_path = os.path.join(logs_dir, f"inference_{clean_game_name}.log")
            
            # Add file handler for inference logs
            self.inference_file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
            self.inference_file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            self.inference_file_handler.setFormatter(formatter)
            
            # Add handler to relevant loggers
            loggers_to_configure = [
                logging.getLogger(__name__),  # agents.rl.rl_agent
                logging.getLogger('agents.rl.arc_grid_env'),
                logging.getLogger('stable_baselines3'),
            ]
            
            for log_obj in loggers_to_configure:
                log_obj.addHandler(self.inference_file_handler)
            
            # Log session start information
            logger.info("=" * 50)
            logger.info(f"🚀 RL INFERENCE SESSION START")
            logger.info(f"Game: {self.game_id}")
            logger.info(f"Agent: {self.agent_name}")
            logger.info(f"Scorecard: {self.card_id}")
            logger.info(f"Tags: {self.tags}")
            logger.info(f"Log file: {log_file_path}")
            
            # Generate scorecard URL for easy access (same as training phase)
            if self.card_id and '.' in self.card_id:
                card_id_for_url = self.card_id.replace('.', '-')
                scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id_for_url}"
                logger.info(f"Scorecard URL: {scorecard_url}")
            else:
                logger.info(f"Scorecard URL: {self.ROOT_URL}/scorecards/{self.card_id}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.warning(f"Could not set up inference logging: {e}")
            self.inference_file_handler = None

    def _cleanup_inference_logging(self) -> None:
        """Clean up inference logging handlers."""
        if hasattr(self, 'inference_file_handler') and self.inference_file_handler:
            try:
                # Log session end
                logger.info("=" * 50) 
                logger.info("🏁 RL INFERENCE SESSION COMPLETE")
                logger.info("=" * 50)
                
                # Remove handler from all loggers
                loggers = [
                    logging.getLogger(__name__),
                    logging.getLogger('agents.rl.arc_grid_env'),
                    logging.getLogger('stable_baselines3'),
                ]
                
                for log_obj in loggers:
                    if self.inference_file_handler in log_obj.handlers:
                        log_obj.removeHandler(self.inference_file_handler)
                
                # Close the file handler
                self.inference_file_handler.close()
                
            except Exception as e:
                logger.warning(f"Error during inference logging cleanup: {e}")

    def cleanup(self, scorecard=None) -> None:
        """Clean up RL agent resources."""
        # Clean up our inference logging first
        self._cleanup_inference_logging()
        
        # Call parent cleanup
        super().cleanup(scorecard)

    @property
    def name(self) -> str:
        return f"{super().name}.rl"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        return any([
            latest_frame.state is GameState.WIN,
            # Stop after game over to avoid endless resets
            latest_frame.state is GameState.GAME_OVER,
        ])

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using the trained PPO model with proper environment stepping."""
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Always reset when game is not started or after game over
            self.env_needs_reset = True
            action = GameAction.RESET
            action.reasoning = "Resetting game to start new episode"
            return action
        
        if self.model is None or self.env is None:
            logger.error("❌ Model or environment not loaded, cannot choose action")
            action = GameAction.RESET
            action.reasoning = "Model/environment not available, resetting"
            return action
        
        try:
            # Reset environment if needed (first action after game reset)
            if self.env_needs_reset:
                logger.debug("Resetting environment...")
                self.current_obs, _ = self.env.reset()
                self.env_needs_reset = False
                logger.debug("Environment reset, got initial observation")
            
            # Get action from trained model using current observation
            # Ensure observation is properly formatted for model prediction
            obs_array = self.current_obs
            if hasattr(obs_array, 'shape'):
                logger.debug(f"Observation shape: {obs_array.shape}")
            
            logger.debug("Calling model.predict...")
            try:
                # Make sure observation is numpy array with correct dtype
                import numpy as np
                if not isinstance(obs_array, np.ndarray):
                    obs_array = np.array(obs_array)
                
                # Ensure consistent dtype (int32 is what the model expects)
                if obs_array.dtype != np.int32:
                    obs_array = obs_array.astype(np.int32)
                    
                logger.debug(f"Observation array: shape={obs_array.shape}, dtype={obs_array.dtype}")
                model_action, _states = self.model.predict(obs_array, deterministic=self.deterministic)
                logger.debug(f"Model returned action: {model_action} (type: {type(model_action)})")
            except Exception as predict_error:
                logger.error(f"Model predict failed: {predict_error}")
                raise predict_error
            
            # Step environment with the predicted action
            logger.debug("Stepping environment...")
            self.current_obs, reward, terminated, truncated, info = self.env.step(model_action)
            logger.debug("Environment step completed")
            done = terminated or truncated
            
            # Convert environment action (0-3) to GameAction (ACTION1-4)
            action_mapping = {
                0: GameAction.ACTION1,
                1: GameAction.ACTION2, 
                2: GameAction.ACTION3,
                3: GameAction.ACTION4
            }
            
            # Convert numpy scalar to regular Python int
            # Handle both scalar numpy arrays and regular integers
            if hasattr(model_action, 'item'):
                env_action = model_action.item()  # Extract scalar from numpy array
            else:
                env_action = int(model_action)
            logger.debug(f"Environment action: {env_action} (type: {type(env_action)}), reward: {reward}, done: {done}")
            
            if env_action in action_mapping:
                action = action_mapping[env_action]
                # Ensure all values are converted to native Python types for JSON serialization
                reward_val = float(reward) if hasattr(reward, 'item') else reward
                action.reasoning = f"PPO model + environment step: {action.value} (reward: {reward_val:.3f})"
                return action
            else:
                logger.warning(f"⚠️ Invalid environment action: {env_action}, defaulting to ACTION1")
                action = GameAction.ACTION1
                action.reasoning = f"Invalid environment output {env_action}, using fallback"
                return action
                
        except Exception as e:
            logger.error(f"❌ Error in RL inference: {e}")
            # Log full traceback for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Reset environment state on error
            self.env_needs_reset = True
            action = GameAction.ACTION1
            action.reasoning = f"RL inference failed: {str(e)}"
            return action