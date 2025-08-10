"""
Dynamic Trajectory Buffer for Quality-Based Rollout Collection

This module implements a dynamic rollout system that triggers optimization updates
based on trajectory quality rather than fixed step counts. It integrates with
the existing reward attribution system to identify meaningful interactions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

# Use module-specific logger for proper subprocess logging
logger = logging.getLogger(__name__)


class DynamicTrajectoryBuffer:
    """
    Collects steps and triggers updates based on trajectory quality.
    Integrates with existing attribution system thresholds.
    
    The buffer tracks meaningful environmental interactions and triggers
    optimization updates when:
    1. Enough meaningful steps have accumulated
    2. Important events occur (level completion, life loss)
    3. Trajectory becomes too long without progress
    """
    
    def __init__(self, N: int = 64):
        """
        Initialize the dynamic trajectory buffer.
        
        Args:
            N: Player size (used to calculate movement threshold)
        """
        self.movement_threshold = 2 * N + 1  # Minimum for env change (129 for N=64)
        self.level_threshold = 500  # Level completion detection
        self.life_threshold = 3000  # Life loss detection
        
        self.meaningful_steps: List[Tuple[Any, float]] = []  # (step_data, env_changes)
        self.total_steps: List[Tuple] = []  # All steps regardless of quality
        
        # Trajectory quality metrics
        self.total_env_changes = 0.0
        self.wall_hits = 0
        self.pure_movements = 0
        self.env_interactions = 0
        
        logger.debug(f"DynamicTrajectoryBuffer initialized: movement_threshold={self.movement_threshold}")
    
    def add_step(self, obs: np.ndarray, action: int, reward: float, done: bool, info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Add step and check if update should trigger.
        
        Args:
            obs: Environment observation
            action: Action taken
            reward: Reward received (after attribution)
            done: Whether episode ended
            info: Step information including total_pixels_changed
            
        Returns:
            Tuple of (should_update, reason) where reason explains why update was triggered
        """
        total_changes = info.get('total_pixels_changed', 0)
        
        # Always track all steps for complete trajectory
        step_data = (obs, action, reward, done, info)
        self.total_steps.append(step_data)
        
        # Update trajectory quality metrics
        if total_changes == 0:
            self.wall_hits += 1
        elif total_changes <= self.movement_threshold:
            self.pure_movements += 1
        else:
            # Environmental interaction detected
            env_changes = total_changes - self.movement_threshold
            self.env_interactions += 1
            self.total_env_changes += env_changes
            
            # Track meaningful steps (environmental interaction)
            self.meaningful_steps.append((step_data, env_changes))
        
        # Immediate update triggers for critical events
        if total_changes >= self.life_threshold:
            # Life loss event - immediate update to capture penalty attribution
            logger.debug(f"Life loss event detected: {total_changes} changes, triggering immediate update")
            return True, "LIFE_LOSS_EVENT"
        
        elif total_changes >= self.level_threshold:
            # Level completion event - immediate update to capture completion bonus
            logger.debug(f"Level completion event detected: {total_changes} changes, triggering immediate update")
            return True, "LEVEL_COMPLETION_EVENT"
        
        # Episode termination - always update to close trajectory
        if done:
            logger.debug(f"Episode ended, triggering update for trajectory completion")
            return True, "EPISODE_END"
        
        
        # No update trigger
        return False, None
    
    def get_trajectory_quality(self) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for the trajectory.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.total_steps:
            return {
                'quality_score': 0.0,
                'meaningful_ratio': 0.0,
                'avg_env_impact': 0.0,
                'total_steps': 0,
                'meaningful_steps': 0,
                'wall_hits': 0,
                'pure_movements': 0,
                'env_interactions': 0
            }
        
        # Calculate meaningful step ratio
        meaningful_ratio = len(self.meaningful_steps) / len(self.total_steps)
        
        # Calculate average environmental impact
        if self.meaningful_steps:
            avg_env_impact = sum(ec for _, ec in self.meaningful_steps) / len(self.meaningful_steps)
        else:
            avg_env_impact = 0.0
        
        # Quality score combines ratio and impact magnitude
        # Higher ratio of meaningful steps + higher average impact = better quality
        quality_score = meaningful_ratio * (1.0 + avg_env_impact * 0.01)
        
        return {
            'quality_score': quality_score,
            'meaningful_ratio': meaningful_ratio,
            'avg_env_impact': avg_env_impact,
            'total_steps': len(self.total_steps),
            'meaningful_steps': len(self.meaningful_steps),
            'wall_hits': self.wall_hits,
            'pure_movements': self.pure_movements,
            'env_interactions': self.env_interactions,
            'total_env_changes': self.total_env_changes
        }
    
    def get_trajectory_data(self, attributed_rewards: Dict[int, float] = None) -> List[Tuple]:
        """
        Get all trajectory step data for PPO rollout buffer with attributed rewards injected.
        
        Args:
            attributed_rewards: Dict mapping step numbers to attributed rewards
        
        Returns:
            List of step tuples (obs, action, reward, done, info) with corrected rewards
        """
        if attributed_rewards is None:
            return self.total_steps.copy()
            
        # Inject attributed rewards into trajectory data
        corrected_steps = []
        step_counter = 1  # Steps are 1-indexed in the environment
        
        for obs, action, original_reward, done, info in self.total_steps:
            # Use attributed reward if available, otherwise keep original
            corrected_reward = attributed_rewards.get(step_counter, original_reward)
            
            if step_counter in attributed_rewards:
                logger.info(f"🎯 REWARD INJECTION: Step {step_counter} reward corrected: "
                           f"{original_reward:.1f} → {corrected_reward:.1f}")
            
            corrected_steps.append((obs, action, corrected_reward, done, info))
            step_counter += 1
            
        return corrected_steps
    
    def reset(self):
        """Reset buffer for new trajectory collection."""
        self.meaningful_steps.clear()
        self.total_steps.clear()
        self.total_env_changes = 0.0
        self.wall_hits = 0
        self.pure_movements = 0
        self.env_interactions = 0
        
        logger.debug("DynamicTrajectoryBuffer reset for new trajectory")


class DynamicRolloutCollector:
    """
    Manages dynamic rollout collection across multiple environments.
    
    This class coordinates multiple DynamicTrajectoryBuffer instances
    and integrates with PPO training to provide quality-based rollouts.
    """
    
    def __init__(self, n_envs: int, N: int = 64):
        """
        Initialize the dynamic rollout collector.
        
        Args:
            n_envs: Number of parallel environments
            N: Player size for movement threshold calculation
        """
        self.n_envs = n_envs
        self.buffers = [DynamicTrajectoryBuffer(N) for _ in range(n_envs)]
        self.collected_trajectories: List[Dict[str, Any]] = []
        
        logger.info(f"DynamicRolloutCollector initialized for {n_envs} environments")
    
    def _distribute_level_completion_rewards(self, trajectory_steps: List[Dict], trajectory_data: List) -> List:
        """
        Distribute level completion rewards equally across all steps in trajectory.
        
        Args:
            trajectory_steps: List of dicts with step info including level_complete flag
            trajectory_data: List of tuples (obs, action, reward, done, info)
            
        Returns:
            Updated trajectory_data with distributed rewards
        """
        # Check if any step completed a level
        level_complete_steps = [s for s in trajectory_steps if s.get('level_complete', False)]
        
        if not level_complete_steps:
            # No level completion, return original data
            return trajectory_data
        
        # Calculate total level completion bonus (sum of pixel changes from level completion steps)
        total_level_bonus = sum(s['base_reward'] for s in level_complete_steps)
        
        # Distribute bonus equally across all steps
        bonus_per_step = total_level_bonus / len(trajectory_data)
        
        logger.info(f"📊 Level completion detected! Distributing {total_level_bonus:.1f} reward "
                   f"across {len(trajectory_data)} steps ({bonus_per_step:.1f} per step)")
        
        # Update rewards in trajectory data
        updated_trajectory = []
        for i, (obs, action, reward, done, info) in enumerate(trajectory_data):
            new_reward = reward + bonus_per_step
            updated_trajectory.append((obs, action, new_reward, done, info))
        
        return updated_trajectory
    
    def collect_rollouts(self, env, model, target_trajectories: int = 4) -> List[Dict[str, Any]]:
        """
        Collect dynamic rollouts based on trajectory quality.
        
        Args:
            env: Vectorized environment
            model: PPO model for action prediction
            target_trajectories: Target number of complete trajectories to collect
            
        Returns:
            List of trajectory dictionaries with steps and quality metrics
        """
        collected_trajectories = []
        
        # Start with current environment observations (no forced reset)
        # Note: We assume environments are already initialized and running
        # The VecEnv should have current observations from previous interactions
        logger.info("Starting dynamic rollout collection with current environment states")
        logger.info(f"Target: {target_trajectories} trajectories")
        
        # Start rollout collection without any environment manipulation
        # We collect trajectories as environments naturally evolve through model actions
        logger.info("Starting dynamic rollout collection from natural environment evolution (no reset, no dummy steps)")
        
        step_count = 0
        obs = None  # Will be set from first step
        
        while len(collected_trajectories) < target_trajectories:
            # For the first iteration, we need to get observations without knowing the current state
            # We'll use random actions for the very first step to bootstrap observations
            if obs is None:
                # First step only - use random actions to get initial observations
                actions = [env.action_space.sample() for _ in range(self.n_envs)]
                logger.info("Using random actions for first step to bootstrap observations")
            else:
                # Normal step - get actions from model
                actions, _ = model.predict(obs, deterministic=False)
            
            # Step environments
            new_obs, rewards, dones, infos = env.step(actions)
            
            # Process each environment
            enough_trajectories = False
            
            for env_idx in range(self.n_envs):
                # For the first step, obs might be None, so use new_obs as both previous and current
                current_obs = new_obs[env_idx] if obs is None else obs[env_idx]
                should_update, reason = self.buffers[env_idx].add_step(
                    current_obs, actions[env_idx], rewards[env_idx], dones[env_idx], infos[env_idx]
                )
                
                if should_update:
                    # Trajectory complete - check for level completion and distribute rewards
                    logger.info(f"🎯 Processing trajectory from env {env_idx}")
                    
                    # Get current trajectory from environment for level completion processing
                    trajectory_steps = None
                    try:
                        # Get the trajectory steps with level completion flags
                        result = env.env_method('get_current_trajectory', indices=[env_idx])
                        if result and result[0]:
                            trajectory_steps = result[0]
                            logger.info(f"🎯 Found {len(trajectory_steps)} steps in trajectory")
                    except (AttributeError, Exception) as e:
                        logger.debug(f"Could not get trajectory from env {env_idx}: {e}")
                    
                    # Extract trajectory data
                    trajectory_data = self.buffers[env_idx].get_trajectory_data(None)  # No attributed rewards
                    quality_metrics = self.buffers[env_idx].get_trajectory_quality()
                    
                    # Apply trajectory-wide reward distribution if level was completed
                    if trajectory_steps and trajectory_data:
                        trajectory_data = self._distribute_level_completion_rewards(trajectory_steps, trajectory_data)
                    
                    # Clear environment's trajectory after processing
                    try:
                        env.env_method('clear_current_trajectory', indices=[env_idx])
                    except (AttributeError, Exception) as e:
                        logger.debug(f"Could not clear trajectory for env {env_idx}: {e}")
                    
                    # Get step range for trajectory logging
                    if trajectory_data:
                        first_step = trajectory_data[0]
                        last_step = trajectory_data[-1]
                        # Extract step numbers from info if available
                        first_step_num = first_step[4].get('steps', 'unknown') if len(first_step) > 4 and hasattr(first_step[4], 'get') else 'unknown'
                        last_step_num = last_step[4].get('steps', 'unknown') if len(last_step) > 4 and hasattr(last_step[4], 'get') else 'unknown'
                        step_range = f"steps {first_step_num}-{last_step_num}" if first_step_num != 'unknown' and last_step_num != 'unknown' else f"{len(trajectory_data)} steps"
                    else:
                        step_range = "0 steps"
                    
                    trajectory = {
                        'env_idx': env_idx,
                        'steps': trajectory_data,
                        'quality_metrics': quality_metrics,
                        'trigger_reason': reason,
                        'step_count': len(trajectory_data)
                    }
                    
                    collected_trajectories.append(trajectory)
                    
                    logger.info(f"📊 DYNAMIC ROLLOUT: Collected trajectory from env {env_idx}: {step_range}, "
                               f"quality={quality_metrics['quality_score']:.3f}, reason={reason}")
                    
                    logger.debug(f"Collected trajectory from env {env_idx}: {len(trajectory_data)} steps, "
                               f"quality={quality_metrics['quality_score']:.3f}, reason={reason}")
                    
                    # Reset buffer for new trajectory
                    self.buffers[env_idx].reset()
                    
                    # FIXED: Set flag to exit both loops completely
                    if len(collected_trajectories) >= target_trajectories:
                        enough_trajectories = True
                        break
            
            # FIXED: Exit while loop if we have enough trajectories
            if enough_trajectories:
                break
            
            # Update observations
            obs = new_obs
            step_count += 1
            
            # Safety check to prevent infinite loops
            if step_count > 10000:  # Adjust based on expected episode length
                logger.warning(f"Dynamic rollout collection exceeded safety limit ({step_count} steps)")
                break
        
        # Log collection summary
        total_steps = sum(t['step_count'] for t in collected_trajectories)
        avg_quality = np.mean([t['quality_metrics']['quality_score'] for t in collected_trajectories])
        
        logger.info(f"Dynamic rollout collection complete: {len(collected_trajectories)} trajectories, "
                   f"{total_steps} total steps, avg_quality={avg_quality:.3f}")
        
        return collected_trajectories
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection state.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {
            'active_buffers': sum(1 for buf in self.buffers if buf.total_steps),
            'total_steps_in_buffers': sum(len(buf.total_steps) for buf in self.buffers),
            'total_meaningful_steps': sum(len(buf.meaningful_steps) for buf in self.buffers),
        }
        
        if stats['total_steps_in_buffers'] > 0:
            stats['overall_meaningful_ratio'] = stats['total_meaningful_steps'] / stats['total_steps_in_buffers']
        else:
            stats['overall_meaningful_ratio'] = 0.0
        
        return stats