"""
ARCGridEnvironment - Natural 64x64 Grid Environment for ARC-AGI-3

This module provides a Gymnasium environment with the natural (64, 64) observation
space, eliminating the need for format conversions in most representations.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from collections import deque

import gymnasium as gym
import numpy as np

from ..structs import GameState
from .config_loader import get_config
from .swarm_operations import SwarmOperations
from .calibration import Calibration

logger = logging.getLogger(__name__)


class ARCGridEnvironment(gym.Env):
    """
    Gymnasium environment with natural 64x64 grid observations.
    
    This environment provides observations in the most natural format for ARC grids:
    a simple 64x64 tensor of color values. This eliminates conversion overhead
    for CNN-based representations like temporal stacking.
    
    Key Differences from ARCSwarmEnvironment:
    - Observation space: (64, 64) instead of (4096, 3)  
    - Direct grid representation, no coordinate features
    - Optimized for spatial CNN architectures
    - GPS+SAGE adapters handle conversion when needed
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 game_id: str,
                 max_steps: Optional[int] = None,
                 tags: Optional[list] = None,
                 env_id: int = 0,
                 scorecard_id: Optional[str] = None,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config_file: Optional[str] = None):
        """
        Initialize ARC Grid Environment with fixed (10, 64, 64) stacked observations.
        
        Args:
            game_id: Game identifier (e.g., "ls20")
            max_steps: Maximum steps per episode (from config if None)
            tags: Tags for scorecard tracking
            env_id: Environment ID for parallel environments
            scorecard_id: Optional existing scorecard ID to use
            root_url: Optional API root URL
            api_key: Optional API key
            config_file: Optional config file to use (defaults to config.json)
        """
        super().__init__()
        
        # Load configuration
        config = get_config(config_file)
        
        # Fixed to PPO-only (no algorithm configuration needed)
        self.game_id = game_id
        self.max_steps = max_steps or config.max_actions
        self.env_id = env_id
        
        # Initialize swarm operations with optional shared resources
        self.swarm_ops = SwarmOperations(
            game_id=game_id, 
            tags=tags,
            scorecard_id=scorecard_id,
            root_url=root_url,
            api_key=api_key
        )
        
        # Fixed observation space: always (10, 64, 64) stacked temporal grids
        self.observation_space = gym.spaces.Box(
            low=0,
            high=15,
            shape=(10, 64, 64),  # Fixed 10-channel temporal stacking
            dtype=np.int32
        )
        
        # Fixed action space based on comprehensive testing results
        # RESET (ACTION0) removed - episodes reset via env.reset() only
        # ACTION5 and ACTION6 confirmed as universal no-ops and excluded
        # ACTION1-4 provide optimal learning signal for puzzle solving
        self.supported_actions = [1, 2, 3, 4]
        
        # Create fixed action space - no dynamic querying needed
        self.action_space = gym.spaces.Discrete(4)  # Actions 1-4 only
        
        # Create mapping from gym action indices to actual action IDs
        self.action_mapping = {i: action_id for i, action_id in enumerate(self.supported_actions)}
        self.reverse_action_mapping = {action_id: i for i, action_id in enumerate(self.supported_actions)}
        
        logger.info(f"Fixed RL action space optimized for puzzle solving:")
        logger.info(f"  • Actions 1-4 included (core puzzle actions)")
        logger.info(f"  • ACTION0 (RESET) excluded (handled by env.reset())")
        logger.info(f"  • ACTION5 excluded (universal no-op)")
        logger.info(f"  • ACTION6 excluded (universal no-op + requires coordinates)")
        logger.info(f"RL action space: Discrete(4) -> {self.action_mapping}")
        
        # Episode state
        self.steps = 0
        self.current_score = 0
        self.previous_api_score = 0  # Track API score to calculate actual deltas
        self.previous_grid = None
        self.action_history = []
        self.episode_reward = 0.0
        
        # Store temporal depth for stacked observations
        self.temporal_depth = 10  # Fixed temporal depth for stacked observations
        
        # Grid history for stacked observations (maintains temporal depth)
        # grid_history[0] = t-1, grid_history[1] = t-2, etc.
        # Store (temporal_depth - 1) historical grids to fully utilize network capacity
        self.max_history_depth = self.temporal_depth - 1  # 9 historical grids for 10-channel network
        self.grid_history = deque(maxlen=self.max_history_depth)
        
        # Initialize current trajectory for level completion tracking
        self.current_trajectory = []
        
        # Initialize player tracker only for engineered mode (still used for rewards in all modes)
        self.player_tracker = Calibration()
        
        # Initialize grid visualization logging (controlled by config)
        self.grid_visualization_enabled = config.grid_visualization_enabled
        if self.grid_visualization_enabled:
            self._init_grid_logging()
        else:
            self.grid_log_dir = None
            self.episode_count = 0
            self.current_episode_dir = None
        
        logger.info(f"ARCGridEnvironment initialized for game {game_id} (env_id={env_id}) with elegant reward system")
        logger.info(f"[ENV{env_id}] Grid visualization: {'ENABLED' if self.grid_visualization_enabled else 'DISABLED'}")
    
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial grid observation of shape (64, 64)
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Increment episode counter (for proper episode tracking)
        self.episode_count += 1
        
        # Reset episode state
        self.steps = 0
        self.current_score = 0
        self.previous_api_score = 0  # Track API score to calculate actual deltas
        self.previous_grid = None
        self.action_history = []
        self.episode_reward = 0.0
        
        # Reset grid history for stacked observations
        self.grid_history = deque(maxlen=self.max_history_depth)
        
        # Reset current trajectory for new episode
        self.current_trajectory = []
        
        # Reset player tracker
        self.player_tracker.reset()
        
        # Reset game via swarm operations
        frame = self.swarm_ops.reset_game()
        
        if frame is None:
            # Handle reset failure gracefully
            logger.warning("Game reset failed, returning zero observation")
            observation = self._get_obs()
            info = self._get_info(error="Reset failed")
        else:
            # Get current grid and associated game response data first
            current_grid = self._frame_to_grid(frame)
            self.current_score = frame.score
            self.previous_api_score = frame.score  # Initialize API score tracking
            
            # Store initial grid for reward calculation (2D grid, not observation)
            self.previous_grid = current_grid.copy()
            
            # Create stacked observation
            observation = self._get_obs(frame)

            # Attempt player calibration
            calibration_success, test_frame = self.player_tracker.calibrate_at_reset(self.swarm_ops)
            
            if calibration_success:
                if test_frame is not None:
                    current_grid = self._frame_to_grid(test_frame)
                    self.previous_grid = current_grid.copy()
                    observation = self._get_obs(test_frame)
                logger.info(f"Calibration successful: player_size={self.player_tracker.player_size}, "
                          f"energy_consumption={self.player_tracker.energy_consumption}")
                logger.info(f"Game continues from the calibrated state")
            
            info = self._get_info(frame, player_calibrated=calibration_success)
            
            # Save initial grid visualization (if enabled)
            if self.grid_visualization_enabled:
                self._save_grid_image(current_grid, step=0, episode_start=True)
            
            # Log episode start
            logger.info(f"[ENV{self.env_id}] ===== EPISODE {self.episode_count} START ===== "
                       f"Game: {self.game_id} | Initial score: {self.current_score} | "
                       f"Player calibrated: {calibration_success}")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Gym action index (0 to len(supported_actions)-1)
            
        Returns:
            observation: New grid observation of shape (64, 64)
            reward: Reward for this step
            terminated: Whether episode is finished
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Normal step processing - duplicate call issue should be fixed in rollout collector
        
        # Map gym action index to actual action ID
        # Convert numpy scalar to Python int if needed
        action_key = int(action.item()) if hasattr(action, 'item') else int(action)
        
        if action_key in self.action_mapping:
            actual_action_id = self.action_mapping[action_key]
        else:
            logger.warning(f"Invalid action {action_key}, using ACTION1")
            actual_action_id = 1  # Default to ACTION1 (no RESET available)
        
        # Track action history BEFORE executing action (use actual action ID)
        self.action_history.append(actual_action_id)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Store previous state for change analysis
        old_grid = self.previous_grid.copy() if self.previous_grid is not None else None
        
        # Execute action via swarm operations using actual action ID
        frame = self.swarm_ops.execute_action(actual_action_id)
        
        if frame is None:
            # Handle action failure
            logger.warning(f"Action {action} (mapped to {actual_action_id}) failed")
            observation = self._get_obs()
            reward = -1.0  # Penalty for failed action
            terminated = True
            truncated = False
            info = self._get_info(error="Action failed")
        else:
            # Calculate actual score delta by comparing with previous API score
            current_api_score = frame.score
            score_delta = current_api_score - self.previous_api_score
            current_grid = self._frame_to_grid(frame)
            
            # Perform change analysis for logging and engineered observations (optional for reward)
            if old_grid is not None:
                change_analysis = self.player_tracker.separate_changes(old_grid, current_grid, actual_action_id)
            else:
                # First step - no change analysis possible
                change_analysis = {
                    'total_pixels_changed': 0,
                    'player_movement_pixels': 0,
                    'environment_interaction_pixels': 0,
                    'environment_change_mask': np.zeros_like(current_grid, dtype=bool),
                    'player_movement_mask': np.zeros_like(current_grid, dtype=bool),
                    'calibrated': self.player_tracker.calibrated
                }
            
            # Calculate actual values used by arithmetic reward system for logging
            if self.player_tracker.calibrated and old_grid is not None:
                actual_total_changes = np.sum(current_grid != old_grid)
                N = self.player_tracker.player_size
                e = self.player_tracker.energy_consumption
                
                # Wall collision: no movement, no energy consumption
                if actual_total_changes == 0:
                    actual_player_movement = 0
                    actual_energy_consumption = 0
                    actual_pure_env_changes = 0
                else:
                    # Normal movement: theoretical values apply
                    actual_player_movement = 2 * N  # Expected player movement pixels
                    actual_energy_consumption = e   # Expected energy consumption pixels
                    actual_pure_env_changes = actual_total_changes - (actual_player_movement + actual_energy_consumption)
                
            else:
                actual_total_changes = change_analysis['total_pixels_changed']
                actual_player_movement = change_analysis['player_movement_pixels']
                actual_energy_consumption = self.player_tracker.energy_consumption if self.player_tracker.energy_consumption else 1
                actual_pure_env_changes = "N/A"
            
            # Calculate reward and check for level completion
            reward, level_complete = self._calculate_reward(score_delta, frame.state, actual_action_id, current_grid)
            
            # Store level completion flag in current trajectory
            if not hasattr(self, 'current_trajectory'):
                self.current_trajectory = []
            
            self.current_trajectory.append({
                'step': self.steps,
                'action': actual_action_id,
                'base_reward': reward,
                'level_complete': level_complete
            })
            
            # IMPORTANT: Store terminal observation BEFORE updating state
            terminal_observation = self._get_obs(frame)
            
            # Save grid visualization for this step (if enabled)
            if self.grid_visualization_enabled:
                logger.debug(f"🖼️ [ENV{self.env_id}] SAVING VISUALIZATION: step={self.steps + 1}, action_id={actual_action_id}")
                self._save_grid_image(current_grid, step=self.steps + 1, action_id=actual_action_id, reward=reward)
            
            # Update state - track cumulative level completions
            if score_delta > 0:  # Level completed
                self.current_score += 1  # Increment cumulative level count
                
            # Update grid history for stacked observations (before updating previous_grid)
            if self.previous_grid is not None:
                # Add previous grid to history (most recent first)
                # deque with maxlen automatically maintains size limit
                self.grid_history.appendleft(self.previous_grid.copy())
                    
            self.previous_grid = current_grid
            self.previous_api_score = current_api_score  # Update API score tracking
            self.steps += 1
            self.episode_reward += reward
            
            # Log detailed action execution with raw pixel reward values (AFTER state update)
            logger.info(f"[ENV{self.env_id}] Episode {self.episode_count} Step {self.steps}: "
                       f"gym_action={action} -> ACTION{actual_action_id} | "
                       f"reward={reward:.1f} (raw_pixels, score_delta={score_delta}) | "
                       f"state={frame.state.value} | "
                       f"level={self.current_score} | "
                       f"api_score={current_api_score} (prev={self.previous_api_score - score_delta}) | "
                       f"total_changes={actual_total_changes} | "
                       f"player_movement={actual_player_movement} | "
                       f"energy_consumption={actual_energy_consumption} | "
                       f"pure_env_changes={actual_pure_env_changes} | "
                       f"calibrated={self.player_tracker.calibrated} | "
                       f"episode_reward={self.episode_reward:.1f}")
            
            # Check termination conditions - only terminate on WIN or GAME_OVER from API
            if frame.state == GameState.WIN:
                # All levels complete - episode ends (API determined)
                terminated = True
                logger.info(f"[ENV{self.env_id}] WIN - All levels complete! Final score: {frame.score}")
            elif frame.state == GameState.GAME_OVER:
                # Energy depleted or failed - episode ends (API determined)
                terminated = True
                logger.info(f"[ENV{self.env_id}] GAME OVER - Player failed or energy depleted")
            else:
                # Continue episode - let API handle level transitions naturally
                terminated = False
                
                # Log level completion for tracking but don't terminate episode
                if score_delta > 0:
                    logger.info(f"[ENV{self.env_id}] Level complete! Score: {frame.score} (continuing episode)")
            
            # Check for max steps truncation
            truncated = self.steps >= self.max_steps
            
            # Log episode end if terminated or truncated
            if terminated or truncated:
                end_reason = "ALL_LEVELS_COMPLETE" if (frame.state == GameState.WIN and self.current_score >= 6) else \
                           "GAME_OVER" if frame.state == GameState.GAME_OVER else \
                           "MAX_STEPS"
                logger.info(f"[ENV{self.env_id}] ===== EPISODE {self.episode_count} END ===== "
                           f"Reason: {end_reason} | Final score: {self.current_score} | "
                           f"Total steps: {self.steps} | Episode reward: {self.episode_reward + reward:.3f}")
            
            # Get base info including change analysis and level completion flag
            info = self._get_info(frame, score_delta, change_analysis=change_analysis)
            info['level_complete'] = level_complete
            
            # Return the terminal observation - VecEnv will handle reset and terminal_observation storage
            observation = terminal_observation
        
        # Store result for duplicate detection
        result = (observation, reward, terminated, truncated, info)
        self._last_step_result = result
        
        return result
    
    def _frame_to_grid(self, frame_data) -> np.ndarray:
        """
        Convert frame data to 64x64 grid observation.
        
        Args:
            frame_data: Frame data from swarm operations
            
        Returns:
            Grid observation of shape (64, 64)
        """
        if frame_data is None or not frame_data.frame:
            return np.zeros((64, 64), dtype=np.int32)
        
        # Extract grid from frame data
        grid_data = frame_data.frame
        
        # Handle different grid formats
        if isinstance(grid_data, list):
            # Convert list to numpy array
            grid = np.array(grid_data, dtype=np.int32)
            
            # If it's a 3D array (frame format), extract the first frame
            if len(grid.shape) == 3:
                # Assuming shape is (1, 64, 64) or similar
                if grid.shape[0] == 1:
                    grid = grid[0]
                else:
                    # Take the first 2D slice
                    grid = grid[0]
                    
        elif isinstance(grid_data, np.ndarray):
            grid = grid_data.astype(np.int32)
        else:
            # Fallback: empty grid
            return np.zeros((64, 64), dtype=np.int32)
        
        # Ensure grid is 64x64
        if grid.shape != (64, 64):
            # Handle different shapes by padding or cropping to 64x64
            padded_grid = np.zeros((64, 64), dtype=np.int32)
            min_h = min(grid.shape[0], 64)
            min_w = min(grid.shape[1] if len(grid.shape) > 1 else 1, 64)
            
            if len(grid.shape) == 1:
                # 1D array - reshape if possible
                if grid.shape[0] == 4096:
                    grid = grid.reshape(64, 64)
                else:
                    # Can't reshape - use empty grid
                    return padded_grid
            
            padded_grid[:min_h, :min_w] = grid[:min_h, :min_w]
            grid = padded_grid
        
        return grid
    
    def _get_obs(self, frame=None) -> np.ndarray:
        """
        Get observation from current environment state.
        
        Args:
            frame: Optional frame data. If None, creates zero observation.
            
        Returns:
            Fixed (10, 64, 64) stacked temporal observation for PPO
        """
        if frame is None:
            return self._create_zero_observation()
        
        # Get current grid
        current_grid = self._frame_to_grid(frame)
        
        # Always create stacked temporal observation (10, 64, 64)
        observation = self.player_tracker.create_stacked_observation(
            current_grid=current_grid,
            grid_history=self.grid_history,
            temporal_depth=self.temporal_depth
        )
        
        return observation
    
    def _get_info(self, frame=None, score_delta=0, error=None, change_analysis=None, **kwargs) -> Dict[str, Any]:
        """
        Get auxiliary information about current environment state.
        
        Args:
            frame: Optional frame data
            score_delta: Change in score for this step
            error: Optional error message
            
        Returns:
            Dictionary with environment information
        """
        base_info = {
            "score": self.current_score,
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "max_steps": self.max_steps,
            "env_id": self.env_id,
            "guid": self.swarm_ops.guid,
            "game_id": self.game_id
        }
        
        if frame:
            base_info.update({
                "state": frame.state.value if frame.state else "unknown",
                "score_delta": score_delta
            })
        
        if error:
            base_info["error"] = error
            
        # Add change analysis if provided
        if change_analysis:
            base_info.update({
                "total_pixels_changed": change_analysis['total_pixels_changed'],
                "player_movement_pixels": change_analysis['player_movement_pixels'],
                "environment_interaction_pixels": change_analysis['environment_interaction_pixels'],
                "player_calibrated": change_analysis['calibrated']
            })
            
        # Add player tracker state info
        tracker_info = self.player_tracker.get_state_info()
        base_info.update({
            "player_tracker": tracker_info
        })
        
        # Add any additional kwargs
        base_info.update(kwargs)
            
        return base_info
    
    def _create_zero_observation(self) -> np.ndarray:
        """Create a zero observation for error cases."""
        return np.zeros((10, 64, 64), dtype=np.int32)
    
    def _calculate_reward(self, score_delta: int, state: GameState, action: int, current_grid: Optional[np.ndarray], change_analysis: Optional[Dict] = None) -> Tuple[float, bool]:
        """
        Natural log-scaled pixel-based rewards with level completion flag.
        
        Returns:
            Tuple[float, bool]: (log_scaled_reward, is_level_complete)
        """
        # Check if we have the necessary data
        if current_grid is None or self.previous_grid is None:
            logger.debug("No grid data available for reward calculation")
            return 0.0, False
        
        # Calculate total pixel changes
        total_changes = np.sum(current_grid != self.previous_grid)
        
        # Apply natural log scaling for reward smoothing, but filter out life loss events
        if total_changes == 0:
            log_reward = 0.0
        elif total_changes >= 3000:
            # Large pixel changes likely indicate life loss - give zero reward
            log_reward = 0.0
            logger.debug(f"Life loss detected: {total_changes} pixels → 0.0 reward (no penalty)")
        else:
            log_reward = float(np.log(total_changes + 1))
        
        # Level completion detection
        if score_delta > 0:
            # Return log-scaled reward + flag for trajectory distribution
            logger.info(f"Level completion detected: {total_changes} pixels → {log_reward:.2f} log reward")
            return log_reward, True
        
        # Terminal states - no special handling
        if state in [GameState.WIN, GameState.GAME_OVER]:
            return 0.0, False
        
        # Normal exploration - log-scaled pixel change rewards
        logger.debug(f"Normal step: {total_changes} pixels → {log_reward:.2f} log reward")
        return log_reward, False
    
    def render(self, mode: str = "human"):
        """Render the environment (placeholder)."""
        if mode == "human":
            print(f"Game: {self.game_id}, Steps: {self.steps}, Score: {self.current_score}")
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, 'swarm_ops'):
            self.swarm_ops.cleanup()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current environment state information."""
        info = self.swarm_ops.get_state_info()
        info.update({
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "max_steps": self.max_steps,
            "env_id": self.env_id
        })
        return info
    
    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    
    def _init_grid_logging(self):
        """Initialize grid visualization logging system."""
        import os
        
        # Create logging directory structure
        log_base_dir = "./agents/rl/logs"
        self.grid_log_dir = os.path.join(log_base_dir, f"grid_visualizations_env{self.env_id}")
        os.makedirs(self.grid_log_dir, exist_ok=True)
        
        # Reset episode counter for this environment
        self.episode_count = 0
        self.current_episode_dir = None
        
        logger.info(f"[ENV{self.env_id}] Grid visualization logging enabled: {self.grid_log_dir}")
    
    def _create_arc_colormap(self):
        """Create ARC colormap for visualizations."""
        import matplotlib.colors as mcolors
        
        colors = [
            '#000000',  # 0: black
            '#0074D9',  # 1: blue  
            '#FF4136',  # 2: red
            '#2ECC40',  # 3: green
            '#FFDC00',  # 4: yellow
            '#AAAAAA',  # 5: gray
            '#F012BE',  # 6: magenta
            '#FF851B',  # 7: orange
            '#7FDBFF',  # 8: sky blue
            '#B10DC9',  # 9: purple
            '#FFFF00',  # 10: bright yellow
            '#39CCCC',  # 11: teal
            '#FF69B4',  # 12: pink
            '#87CEEB',  # 13: light blue
            '#FFB6C1',  # 14: light pink
            '#FFFFFF',  # 15: white
        ]
        return mcolors.ListedColormap(colors)
    
    def _save_grid_image(self, grid: np.ndarray, step: int, action_id: Optional[int] = None, 
                        reward: Optional[float] = None, episode_start: bool = False):
        """Save a grid as an image with step information."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues
        import matplotlib.pyplot as plt
        import os
        import threading
        
        # THREAD-SAFE VISUALIZATION DEDUPLICATION
        # Create a unique key for this visualization request
        viz_key = f"ep{self.episode_count}_step{step:03d}_action{action_id if action_id else 'reset'}"
        
        # Initialize visualization tracking if needed
        if not hasattr(self, '_viz_lock'):
            self._viz_lock = threading.Lock()
            self._completed_visualizations = set()
        
        with self._viz_lock:
            # Check if this exact visualization has already been completed
            if viz_key in self._completed_visualizations:
                logger.debug(f"🚫 [ENV{self.env_id}] SKIPPING DUPLICATE VISUALIZATION: {viz_key}")
                return
            
            # Mark this visualization as in-progress to prevent other threads
            self._completed_visualizations.add(viz_key)
        
        # Create episode directory if needed
        if self.current_episode_dir is None or episode_start:
            self.current_episode_dir = os.path.join(self.grid_log_dir, f"episode_{self.episode_count:03d}")
            os.makedirs(self.current_episode_dir, exist_ok=True)
        
        # Create clean filename without timestamp (duplicates prevented by lock)
        if episode_start:
            filename = f"step_{step:03d}_reset.png"
            title = f"ENV{self.env_id} Episode {self.episode_count} | Step {step}: RESET"
        else:
            filename = f"step_{step:03d}_action{action_id}.png"
            title = f"ENV{self.env_id} Episode {self.episode_count} | Step {step}: ACTION{action_id}"
            if reward is not None:
                title += f" | Reward: {reward:.1f}"
        
        filepath = os.path.join(self.current_episode_dir, filename)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        cmap = self._create_arc_colormap()
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=15, interpolation='nearest')
        
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        
        # Add subtle grid
        ax.set_xticks(np.arange(-0.5, 64, 8), minor=False)
        ax.set_yticks(np.arange(-0.5, 64, 8), minor=False)
        ax.grid(True, which='major', color='gray', linewidth=0.3, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Color Value', rotation=270, labelpad=15)
        cbar.set_ticks(range(16))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log the save (only for first few steps to avoid spam)
        if step <= 5 or step % 10 == 0:
            logger.debug(f"[ENV{self.env_id}] Saved grid visualization: {filename}")
    
    def _apply_delayed_reward_attribution_REMOVED(self, initial_reward: float, action_id: int, 
                                        total_changes: int, pure_env_changes: float, score_delta: int = 0) -> Tuple[float, str]:
        """
        Apply delayed reward attribution based on score_delta for level completion and pixel_changes for life loss.
        
        Rules:
        - score_delta > 0: Level completion - attribute reward to previous step using pixel changes
        - total_changes ≥ 3000: Life loss - check for consecutive big changes
        - Otherwise: Normal reward (no attribution needed)
        
        Args:
            initial_reward: The reward calculated by normal system
            action_id: Action taken this step
            total_changes: Total pixel changes this step
            pure_env_changes: Pure environmental changes this step
            score_delta: Score change from API (positive = level completion)
            
        Returns:
            Tuple of (final_reward, event_type)
        """
        # Add current step to buffer
        current_step = self.steps + 1  # +1 because we haven't updated self.steps yet
        self.step_buffer.append([
            current_step, action_id, total_changes, pure_env_changes, initial_reward, initial_reward
        ])
        
        # Apply attribution rules based on score_delta detection and timing
        # Key insight: score_delta > 0 at step N means level completion happened at step N (current step)
        # Next step will show big pixel changes from new level that should be attributed back to current step
        
        if score_delta > 0:
            # Level completion detected at current step - mark for delayed attribution from next step
            event_type = "LEVEL_COMPLETION_DETECTED"
            self._pending_level_completion_step = current_step
            
            # Current step gets its normal reward (the level completion action reward)
            final_reward = initial_reward
            logger.info(f"[ENV{self.env_id}] 🔄 LEVEL_COMPLETION_DETECTED: Step {current_step} completed level, "
                       f"gets normal reward {final_reward:.1f}, waiting for next step's pixel changes")
            
        # Check if previous step had level completion and current step should provide attribution
        # Check if current step should get zero reward due to life reset
        elif hasattr(self, '_pending_life_reset') and self._pending_life_reset:
            event_type = "LIFE_RESET"
            final_reward = 0.0
            logger.info(f"[ENV{self.env_id}] 🔄 LIFE_RESET: Step {current_step} gets 0 reward (life respawn/reset)")
            delattr(self, '_pending_life_reset')
            
        # Check if previous step had level completion and current step should provide attribution
        elif hasattr(self, '_pending_level_completion_step') and len(self.step_buffer) >= 2:
            prev_step_entry = self.step_buffer[-2]
            prev_step_num = prev_step_entry[0]
            
            if self._pending_level_completion_step == prev_step_num:
                # Previous step completed level, current step has new level pixel changes
                event_type = "LEVEL_COMPLETION_ATTRIBUTION"
                
                # Attribute current step's big pixel changes to previous step (the level completer)
                self.step_buffer[-2][5] += total_changes  # Add current step's pixel changes to previous step
                
                prev_action = self.step_buffer[-2][1]
                new_reward = self.step_buffer[-2][5]
                
                logger.info(f"[ENV{self.env_id}] 🎯 LEVEL_COMPLETION_ATTRIBUTION: Step {prev_step_num} ACTION{prev_action} "
                           f"gets level transition bonus +{total_changes:.1f} from step {current_step} (total reward: {new_reward:.1f})")
                
                # Current step gets 0 reward (new level visualization)
                final_reward = 0.0
                logger.info(f"[ENV{self.env_id}] 🏁 LEVEL_COMPLETION_ATTRIBUTION: Current step {current_step} gets 0 reward (new level visualization)")
                
                # Clear the pending flag
                delattr(self, '_pending_level_completion_step')
            else:
                # Normal step or life loss processing
                if total_changes >= 3000:
                    event_type = "LIFE_LOSS"
                    final_reward = self._handle_life_loss_attribution(current_step, total_changes, initial_reward)
                else:
                    event_type = "NORMAL"
                    final_reward = initial_reward
                    
        elif total_changes >= 3000:
            # Life loss detected based on big pixel changes at current step
            event_type = "LIFE_LOSS"
            final_reward = self._handle_life_loss_attribution(current_step, total_changes, initial_reward)
            
        else:
            # Normal step - no attribution needed (score_delta=0, changes<3000)
            event_type = "NORMAL"
            final_reward = initial_reward
        
        # Update current step's attributed reward
        self.step_buffer[-1][5] = final_reward
        
        # Log step buffer information for debugging
        if len(self.step_buffer) <= 3 or event_type != "NORMAL":
            logger.debug(f"[ENV{self.env_id}] Step buffer: {[s[:3] + [s[5]] for s in self.step_buffer[-3:]]}")
        
        return final_reward, event_type
    
    def _handle_life_loss_attribution_REMOVED(self, current_step: int, total_changes: int, initial_reward: float) -> float:
        """
        Handle life loss attribution with corrected logic.
        
        Life loss logic (corrected):
        - Step N has >3000 pixel changes → Step N is responsible for life loss  
        - Step N-1 gets its usual reward (no penalty)
        - Step N+1 gets zero reward (life reset/respawn)
        
        Args:
            current_step: Current step number
            total_changes: Pixel changes at current step
            initial_reward: Initial reward for current step
            
        Returns:
            Final reward for current step
        """
        logger.info(f"[ENV{self.env_id}] 💀 LIFE_LOSS: Step {current_step} caused life loss with {total_changes} pixel changes")
        
        # Current step (N) gets penalty for causing life loss
        penalty = -self.accumulated_env_changes
        final_reward = penalty
        
        logger.info(f"[ENV{self.env_id}] 💀 LIFE_LOSS: Step {current_step} gets life loss penalty {penalty:.1f}")
        
        # Mark that next step should get zero reward (life reset)
        self._pending_life_reset = True
        
        return final_reward
    
    def get_attributed_rewards_REMOVED(self) -> Dict[int, float]:
        """
        Extract attributed rewards from step buffer for rollout injection.
        
        Returns:
            Dictionary mapping step numbers to their attributed rewards
        """
        attributed_rewards = {}
        for step_entry in self.step_buffer:
            step_num, action_id, total_changes, pure_env_changes, initial_reward, attributed_reward = step_entry
            # Only include steps where attribution changed the reward
            if attributed_reward != initial_reward:
                attributed_rewards[step_num] = attributed_reward
                
        return attributed_rewards
    
    def clear_step_buffer_REMOVED(self):
        """Clear the step buffer after rewards have been transferred to rollout."""
        self.step_buffer.clear()
    
    def get_current_trajectory(self) -> list:
        """Get the current trajectory for reward distribution."""
        return self.current_trajectory.copy()
    
    def clear_current_trajectory(self):
        """Clear the current trajectory after processing."""
        self.current_trajectory = []
