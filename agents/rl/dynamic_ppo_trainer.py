"""
Dynamic PPO Trainer with Quality-Based Rollouts

This module implements a custom PPO training loop that uses dynamic rollout collection
based on trajectory quality rather than fixed step counts. It integrates with the
existing reward attribution system and trajectory-aware rewards.
"""

import numpy as np
import torch as th
from typing import Dict, Any, List, Optional
import logging
import time
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from agents.rl.dynamic_trajectory_buffer import DynamicRolloutCollector

logger = logging.getLogger(__name__)


class DynamicPPOTrainer:
    """
    Custom PPO trainer that uses dynamic rollout collection.
    
    This trainer replaces the standard fixed-step rollout collection with
    a quality-based system that triggers updates based on meaningful
    environmental interactions and important game events.
    """
    
    def __init__(
        self, 
        model: PPO, 
        env: VecEnv,
        target_trajectories_per_update: int = 4,
        max_rollout_steps: int = 1000
    ):
        """
        Initialize the dynamic PPO trainer.
        
        Args:
            model: PPO model to train
            env: Vectorized environment
            target_trajectories_per_update: Target trajectories to collect per update
            max_rollout_steps: Maximum steps to prevent infinite rollouts
        """
        self.model = model
        self.env = env
        self.n_envs = env.num_envs
        
        # Dynamic rollout configuration
        self.target_trajectories_per_update = target_trajectories_per_update
        self.max_rollout_steps = max_rollout_steps
        
        # Initialize dynamic rollout collector
        self.rollout_collector = DynamicRolloutCollector(
            n_envs=self.n_envs,
            N=64  # Player size
        )
        
        # Training statistics
        self.total_timesteps = 0
        self.update_count = 0
        self.trajectory_qualities = []
        self.collection_stats = []
        
        # Checkpoint callback for automatic saving
        self.checkpoint_callback = None
        
        
        logger.info(f"DynamicPPOTrainer initialized:")
        logger.info(f"  - Environments: {self.n_envs}")
        logger.info(f"  - Target trajectories per update: {target_trajectories_per_update}")
        logger.info(f"  - Max rollout steps: {max_rollout_steps}")
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        tb_log_name: str = "dynamic_ppo"
    ) -> "DynamicPPOTrainer":
        """
        Learn policy using dynamic rollout collection.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            callback: Callback to run during training
            log_interval: How often to log training progress
            tb_log_name: TensorBoard log name
            
        Returns:
            Self for chaining
        """
        logger.info(f"Starting dynamic PPO training for {total_timesteps} timesteps")
        start_time = time.time()
        
        # Initialize callbacks with proper setup
        if callback is not None:
            if isinstance(callback, list):
                for cb in callback:
                    cb.init_callback(self.model)
                    # Set initial training state for callbacks that need it
                    if hasattr(cb, 'on_training_start'):
                        cb.on_training_start(locals(), globals())
            else:
                callback.init_callback(self.model)
                # Set initial training state for callbacks that need it  
                if hasattr(callback, 'on_training_start'):
                    callback.on_training_start(locals(), globals())
        
        # Training loop
        while self.total_timesteps < total_timesteps:
            update_number = self.update_count + 1
            logger.info(f"🔄 TRACE: Starting training loop iteration {update_number}")
            
            # Collect dynamic rollouts
            logger.info("🔄 TRACE: About to collect dynamic rollouts")
            trajectories = self._collect_dynamic_rollouts()
            logger.info("🔄 TRACE: Dynamic rollouts collected successfully")
            
            # FIXED: Notify callbacks about rollout collection
            if callback is not None:
                if isinstance(callback, list):
                    for cb in callback:
                        if hasattr(cb, 'on_rollout_end'):
                            cb.on_rollout_end()
                else:
                    if hasattr(callback, 'on_rollout_end'):
                        callback.on_rollout_end()
            
            # Convert trajectories to rollout buffer format
            logger.info("🔄 TRACE: About to convert trajectories to rollout buffer")
            rollout_data = self._trajectories_to_rollout_buffer(trajectories)
            logger.info("🔄 TRACE: Trajectories converted to rollout buffer successfully")
            
            # Perform PPO update
            logger.info("🔄 TRACE: About to perform PPO update")
            training_metrics = self._update_model(rollout_data)
            logger.info("🔄 TRACE: PPO update completed successfully")
            
            # Update statistics
            self.update_count += 1
            timesteps_this_update = sum(t['step_count'] for t in trajectories)
            self.total_timesteps += timesteps_this_update
            
            # CRITICAL: Update model's timestep counter for SB3 logging and checkpoints
            self.model.num_timesteps = self.total_timesteps
            
            # Log device utilization if using GPU
            if self.model.device.type == 'cuda':
                gpu_memory_used = th.cuda.memory_allocated() / 1e9
                gpu_memory_total = th.cuda.get_device_properties(0).total_memory / 1e9
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                logger.info(f"🖥️ GPU Memory: {gpu_memory_used:.2f}/{gpu_memory_total:.2f} GB ({gpu_memory_percent:.1f}%)")
            
            # Log SB3-style training metrics
            self._log_sb3_training_metrics(training_metrics, trajectories, timesteps_this_update)
            
            # Save checkpoint at regular intervals
            if hasattr(self, 'checkpoint_callback') and self.checkpoint_callback:
                self.checkpoint_callback(self.update_count)
            
            # Log progress
            if self.update_count % log_interval == 0:
                self._log_training_progress(trajectories)
            
            # FIXED: Properly update callbacks with training state before calling on_step
            if callback is not None:
                # Update callback state with current training progress
                callback_locals = {
                    'self': self,
                    'timesteps_this_update': timesteps_this_update,
                    'total_timesteps': self.total_timesteps,
                    'update_count': self.update_count,
                    'training_metrics': training_metrics,
                    'trajectories': trajectories
                }
                
                if isinstance(callback, list):
                    continue_training = True
                    for cb in callback:
                        # Update callback with current state
                        if hasattr(cb, 'update_locals'):
                            cb.update_locals(callback_locals)
                        
                        # Call on_step with proper state
                        if not cb.on_step():
                            continue_training = False
                            break
                    if not continue_training:
                        break
                else:
                    # Update callback with current state
                    if hasattr(callback, 'update_locals'):
                        callback.update_locals(callback_locals)
                        
                    # Call on_step with proper state
                    if not callback.on_step():
                        break
            
            # Check if we've reached the target timesteps
            if self.total_timesteps >= total_timesteps:
                break
        
        training_time = time.time() - start_time
        logger.info(f"Dynamic PPO training completed in {training_time:.2f} seconds")
        logger.info(f"Total updates: {self.update_count}")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        
        # FIXED: Properly finalize callbacks
        if callback is not None:
            if isinstance(callback, list):
                for cb in callback:
                    if hasattr(cb, 'on_training_end'):
                        cb.on_training_end()
            else:
                if hasattr(callback, 'on_training_end'):
                    callback.on_training_end()
        
        return self
    
    def _collect_dynamic_rollouts(self) -> List[Dict[str, Any]]:
        """
        Collect rollouts using dynamic trajectory buffer.
        
        Returns:
            List of trajectory dictionaries
        """
        logger.info(f"🔄 TRACE: Starting _collect_dynamic_rollouts (update {self.update_count + 1})")
        
        # Collect trajectories using dynamic collector
        logger.info("🔄 TRACE: Calling rollout_collector.collect_rollouts()")
        trajectories = self.rollout_collector.collect_rollouts(
            env=self.env,
            model=self.model,
            target_trajectories=self.target_trajectories_per_update
        )
        logger.info(f"🔄 TRACE: collect_rollouts() completed successfully, got {len(trajectories)} trajectories")
        
        # Store quality metrics for analysis
        qualities = [t['quality_metrics']['quality_score'] for t in trajectories]
        self.trajectory_qualities.extend(qualities)
        logger.info(f"🔄 TRACE: Stored quality metrics: {qualities}")
        
        # Store collection statistics
        collection_stats = self.rollout_collector.get_collection_stats()
        self.collection_stats.append(collection_stats)
        logger.info(f"🔄 TRACE: _collect_dynamic_rollouts completed successfully")
        
        return trajectories
    
    def _trajectories_to_rollout_buffer(self, trajectories: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Convert trajectory data to format compatible with PPO rollout buffer.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Dictionary with rollout data arrays
        """
        logger.info(f"🔄 TRACE: Starting _trajectories_to_rollout_buffer with {len(trajectories)} trajectories")
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        
        # Process each trajectory
        for trajectory in trajectories:
            steps = trajectory['steps']
            
            # Extract step data
            for step_idx, (obs, action, reward, done, info) in enumerate(steps):
                all_observations.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done)
                
                # Get value and log_prob from current policy
                obs_tensor = obs_as_tensor(obs[np.newaxis], self.model.device)
                action_tensor = th.tensor([action], device=self.model.device)
                
                with th.no_grad():
                    # Standard PPO evaluation
                    try:
                        values, log_probs, _ = self.model.policy.evaluate_actions(
                                obs_tensor, action_tensor
                            )
                        values = values.cpu().numpy()[0]
                        log_probs = log_probs.cpu().numpy()[0]
                    except Exception as e:
                        logger.error(f"🔄 TRACE: ERROR in Standard PPO processing at step {step_idx}: {e}")
                        raise
                    
                    all_values.append(values)
                    all_log_probs.append(log_probs)
        
        # Convert to numpy arrays
        logger.info(f"🔄 TRACE: About to convert to numpy arrays - {len(all_observations)} steps")
        rollout_data = {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'dones': np.array(all_dones),
            'values': np.array(all_values),
            'log_probs': np.array(all_log_probs)
        }
        logger.info(f"🔄 TRACE: Numpy arrays created successfully")
        
        logger.info(f"🔄 TRACE: Converted {len(trajectories)} trajectories to rollout buffer: "
                    f"{len(all_observations)} total steps")
        logger.info(f"🔄 TRACE: _trajectories_to_rollout_buffer completed successfully")
        
        return rollout_data
    
    def _update_model(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update the PPO model using the collected rollout data.
        
        Args:
            rollout_data: Dictionary with rollout data arrays
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"🔄 TRACE: Starting _update_model with {len(rollout_data['observations'])} steps")
        
        # Standard PPO custom training logic
        logger.info("🔄 TRACE: Standard PPO detected - using custom advantage calculation")
        
        # Calculate advantages using GAE
        logger.info("🔄 TRACE: About to calculate advantages using GAE")
        advantages = self._calculate_advantages(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate returns (advantages + values should have same shape)
        # Ensure values are flattened to match advantages shape
        values_flat = np.array(rollout_data['values']).flatten()
        returns = advantages + values_flat
        
        # Returns and advantages shapes are now properly aligned
        
        # Prepare data for PPO update
        rollout_buffer_data = {
            'observations': rollout_data['observations'],
            'actions': rollout_data['actions'],
            'old_values': values_flat,  # Use flattened values
            'old_log_prob': rollout_data['log_probs'],
            'advantages': advantages,
            'returns': returns
        }
        
        # Perform PPO update using the model's internal method
        training_metrics = self._ppo_update(rollout_buffer_data)
        
        logger.info(f"🔄 TRACE: PPO update completed for {len(rollout_data['observations'])} steps")
        
        return training_metrics
    
    def _calculate_advantages(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        Calculate advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Array of advantage estimates
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # Bootstrap value for last step
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        return advantages
    
    def _ppo_update(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform PPO policy update.
        
        Args:
            rollout_data: Dictionary with processed rollout data
            
        Returns:
            Dictionary with training metrics
        """
        # Convert to tensors with optimized GPU transfer
        use_gpu = self.model.device.type == 'cuda'
        
        # For GPU, create tensors on CPU first with pinned memory, then transfer
        if use_gpu:
            observations = obs_as_tensor(rollout_data['observations'], 'cpu').pin_memory()
            observations = observations.to(self.model.device, non_blocking=True)
            
            actions = th.tensor(rollout_data['actions'], dtype=th.long, device='cpu').pin_memory().to(self.model.device, non_blocking=True).flatten()
            old_log_prob = th.tensor(rollout_data['old_log_prob'], device='cpu').pin_memory().to(self.model.device, non_blocking=True).flatten()
            advantages = th.tensor(rollout_data['advantages'], device='cpu').pin_memory().to(self.model.device, non_blocking=True).flatten()
            returns = th.tensor(rollout_data['returns'], device='cpu').pin_memory().to(self.model.device, non_blocking=True).flatten()
            old_values = th.tensor(rollout_data['old_values'], device='cpu').pin_memory().to(self.model.device, non_blocking=True).flatten()
        else:
            # For CPU, direct tensor creation
            observations = obs_as_tensor(rollout_data['observations'], self.model.device)
            actions = th.tensor(rollout_data['actions'], device=self.model.device, dtype=th.long).flatten()
            old_log_prob = th.tensor(rollout_data['old_log_prob'], device=self.model.device).flatten()
            advantages = th.tensor(rollout_data['advantages'], device=self.model.device).flatten()
            returns = th.tensor(rollout_data['returns'], device=self.model.device).flatten()
            old_values = th.tensor(rollout_data['old_values'], device=self.model.device).flatten()
        
        # Tensor shapes are now correct after fixing the returns calculation
        
        # PPO update epochs
        for epoch in range(self.model.n_epochs):
            # Standard PPO policy evaluation
            values, log_prob, entropy = self.model.policy.evaluate_actions(observations, actions)
            
            values = values.flatten()
            log_prob = log_prob.flatten()  # Ensure log_prob is also flattened
            
            # Calculate policy loss
            log_ratio = log_prob - old_log_prob
            ratio = th.exp(log_ratio)
            
            # Get current clip range value (handle FloatSchedule)
            if hasattr(self.model.clip_range, '__call__'):
                # Safe division to avoid division by zero
                progress = self.model.num_timesteps / max(self.model._total_timesteps, 1)
                current_clip_range = self.model.clip_range(progress)
            else:
                current_clip_range = self.model.clip_range
            
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            
            # Calculate value loss
            if self.model.clip_range_vf is not None:
                # Get current value clip range (handle FloatSchedule)
                if hasattr(self.model.clip_range_vf, '__call__'):
                    # Safe division to avoid division by zero
                    progress = self.model.num_timesteps / max(self.model._total_timesteps, 1)
                    current_clip_range_vf = self.model.clip_range_vf(progress)
                else:
                    current_clip_range_vf = self.model.clip_range_vf
                
                # Clipped value loss
                values_clipped = old_values + th.clamp(
                    values - old_values, -current_clip_range_vf, current_clip_range_vf
                )
                value_loss_1 = (values - returns) ** 2
                value_loss_2 = (values_clipped - returns) ** 2
                value_loss = 0.5 * th.max(value_loss_1, value_loss_2).mean()
            else:
                value_loss = 0.5 * ((values - returns) ** 2).mean()
            
            # Calculate entropy loss
            entropy_loss = -th.mean(entropy)
            
            # Combined loss
            loss = policy_loss + self.model.vf_coef * value_loss + self.model.ent_coef * entropy_loss
            
            # Let SB3's standard logging handle the metrics automatically
            # The model's logger is already configured to write to our main log file
            
            # Optimize
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
            
            self.model.policy.optimizer.step()
        
        # Collect training metrics for SB3-style logging
        with th.no_grad():
            # Calculate final metrics after all epochs
            values, log_prob, entropy = self.model.policy.evaluate_actions(observations, actions)
            values = values.flatten()
            log_prob = log_prob.flatten()
            
            # Final loss calculations for logging
            log_ratio = log_prob - old_log_prob
            ratio = th.exp(log_ratio)
            
            # Get current clip range value (handle FloatSchedule)
            if hasattr(self.model.clip_range, '__call__'):
                progress = self.model.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                current_clip_range = self.model.clip_range(progress)
            else:
                current_clip_range = self.model.clip_range
            
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
            final_policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            
            # Value loss
            if self.model.clip_range_vf is not None:
                if hasattr(self.model.clip_range_vf, '__call__'):
                    progress = self.model.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                    current_clip_range_vf = self.model.clip_range_vf(progress)
                else:
                    current_clip_range_vf = self.model.clip_range_vf
                
                values_clipped = old_values + th.clamp(
                    values - old_values, -current_clip_range_vf, current_clip_range_vf
                )
                value_loss_1 = (values - returns) ** 2
                value_loss_2 = (values_clipped - returns) ** 2
                final_value_loss = 0.5 * th.max(value_loss_1, value_loss_2).mean()
            else:
                final_value_loss = 0.5 * ((values - returns) ** 2).mean()
            
            # Entropy loss
            final_entropy_loss = -th.mean(entropy)
            
            # Combined loss
            final_loss = final_policy_loss + self.model.vf_coef * final_value_loss + self.model.ent_coef * final_entropy_loss
            
            # Calculate additional metrics
            clip_fraction = th.mean((th.abs(ratio - 1) > current_clip_range).float())
            
            # Calculate approximate KL divergence
            approx_kl = th.mean(log_ratio - (ratio - 1))
            
        # Return SB3-style training metrics
        return {
            'train/policy_gradient_loss': final_policy_loss.item(),
            'train/value_loss': final_value_loss.item(),
            'train/entropy_loss': final_entropy_loss.item(),
            'train/loss': final_loss.item(),
            'train/clip_fraction': clip_fraction.item(),
            'train/clip_range': current_clip_range,
            'train/learning_rate': self.model.learning_rate,
            'train/approx_kl': approx_kl.item()
        }
    
    def _log_training_progress(self, trajectories: List[Dict[str, Any]]):
        """
        Log training progress and trajectory quality metrics.
        
        Args:
            trajectories: List of recently collected trajectories
        """
        # Calculate trajectory statistics
        total_steps = sum(t['step_count'] for t in trajectories)
        avg_quality = np.mean([t['quality_metrics']['quality_score'] for t in trajectories])
        avg_meaningful_ratio = np.mean([t['quality_metrics']['meaningful_ratio'] for t in trajectories])
        
        # Trigger reason distribution
        trigger_reasons = [t['trigger_reason'] for t in trajectories]
        reason_counts = {reason: trigger_reasons.count(reason) for reason in set(trigger_reasons)}
        
        logger.info(f"=== Dynamic PPO Update {self.update_count} ===")
        logger.info(f"Timesteps: {self.total_timesteps}")
        logger.info(f"Trajectories collected: {len(trajectories)} ({total_steps} steps)")
        logger.info(f"Average quality score: {avg_quality:.3f}")
        logger.info(f"Average meaningful ratio: {avg_meaningful_ratio:.3f}")
        logger.info(f"Trigger reasons: {reason_counts}")
        
        # Log recent quality trend
        if len(self.trajectory_qualities) >= 20:
            recent_quality = np.mean(self.trajectory_qualities[-20:])
            logger.info(f"Recent quality trend (last 20): {recent_quality:.3f}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        if not self.trajectory_qualities:
            return {
                'total_timesteps': self.total_timesteps,
                'update_count': self.update_count,
                'avg_quality': 0.0,
                'quality_trend': []
            }
        
        return {
            'total_timesteps': self.total_timesteps,
            'update_count': self.update_count,
            'avg_quality': np.mean(self.trajectory_qualities),
            'quality_trend': self.trajectory_qualities[-50:],  # Last 50 trajectory qualities
            'recent_collection_stats': self.collection_stats[-10:] if self.collection_stats else []
        }
    
    def _log_sb3_training_metrics(
        self, 
        training_metrics: Dict[str, float], 
        trajectories: List[Dict[str, Any]], 
        timesteps_this_update: int
    ) -> None:
        """
        Log training metrics to SB3's logger system for CSV/JSON export.
        
        Args:
            training_metrics: Training metrics from PPO update
            trajectories: Collected trajectories for this update
            timesteps_this_update: Number of timesteps processed in this update
        """
        if not hasattr(self.model, 'logger') or self.model.logger is None:
            logger.warning("Model has no logger configured - metrics will not be saved to CSV/JSON")
            return
        
        # Core training metrics from PPO update
        for key, value in training_metrics.items():
            self.model.logger.record(key, value)
        
        # Rollout and trajectory metrics
        avg_trajectory_quality = np.mean([t['quality_metrics']['quality_score'] for t in trajectories])
        avg_meaningful_ratio = np.mean([t['quality_metrics']['meaningful_ratio'] for t in trajectories])
        total_dynamic_steps = sum(t['step_count'] for t in trajectories)
        
        self.model.logger.record("rollout/ep_len_mean", total_dynamic_steps / len(trajectories))
        self.model.logger.record("rollout/dynamic_trajectories", len(trajectories))
        self.model.logger.record("rollout/dynamic_steps", total_dynamic_steps)
        self.model.logger.record("rollout/avg_trajectory_quality", avg_trajectory_quality)
        self.model.logger.record("rollout/avg_meaningful_ratio", avg_meaningful_ratio)
        
        # Trigger reason distribution
        trigger_reasons = [t['trigger_reason'] for t in trajectories]
        reason_counts = {reason: trigger_reasons.count(reason) for reason in set(trigger_reasons)}
        for reason, count in reason_counts.items():
            self.model.logger.record(f"rollout/trigger_{reason}", count)
        
        # Time and iteration metrics
        self.model.logger.record("time/iterations", self.update_count)
        self.model.logger.record("time/total_timesteps", self.total_timesteps)
        self.model.logger.record("time/timesteps_this_update", timesteps_this_update)
        
        # Dump all recorded metrics to CSV/JSON files
        # This is the critical step that writes data to progress.csv and progress.json
        self.model.logger.dump(step=self.total_timesteps)
        
        logger.info(f"📊 LOGGED METRICS: Recorded {len(training_metrics)} training metrics + rollout data to SB3 logger")
        logger.info(f"📊 METRICS DUMPED: CSV/JSON files updated at timestep {self.total_timesteps}")
        
        # Write clean metrics table to log file
        self._write_metrics_table_to_log(training_metrics, trajectories, timesteps_this_update)
    
    def _write_metrics_table_to_log(
        self, 
        training_metrics: Dict[str, float], 
        trajectories: List[Dict[str, Any]], 
        timesteps_this_update: int
    ) -> None:
        """
        Write a clean formatted metrics table directly to the log.
        
        Args:
            training_metrics: Training metrics from PPO update
            trajectories: Collected trajectories for this update  
            timesteps_this_update: Number of timesteps processed in this update
        """
        # Collect all metrics (same as what we logged to SB3)
        all_metrics = {}
        
        # Add core training metrics
        for key, value in training_metrics.items():
            all_metrics[key] = value
        
        # Add rollout and trajectory metrics
        avg_trajectory_quality = np.mean([t['quality_metrics']['quality_score'] for t in trajectories])
        avg_meaningful_ratio = np.mean([t['quality_metrics']['meaningful_ratio'] for t in trajectories])
        total_dynamic_steps = sum(t['step_count'] for t in trajectories)
        
        all_metrics["rollout/ep_len_mean"] = total_dynamic_steps / len(trajectories)
        all_metrics["rollout/dynamic_trajectories"] = len(trajectories)
        all_metrics["rollout/dynamic_steps"] = total_dynamic_steps
        all_metrics["rollout/avg_trajectory_quality"] = avg_trajectory_quality
        all_metrics["rollout/avg_meaningful_ratio"] = avg_meaningful_ratio
        
        # Add trigger reason distribution
        trigger_reasons = [t['trigger_reason'] for t in trajectories]
        reason_counts = {reason: trigger_reasons.count(reason) for reason in set(trigger_reasons)}
        for reason, count in reason_counts.items():
            all_metrics[f"rollout/trigger_{reason}"] = count
        
        # Add time and iteration metrics
        all_metrics["time/iterations"] = self.update_count
        all_metrics["time/total_timesteps"] = self.total_timesteps
        all_metrics["time/timesteps_this_update"] = timesteps_this_update
        
        # Create formatted table
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"TRAINING METRICS - Update {self.update_count} (Timestep {self.total_timesteps})")
        lines.append("=" * 60)
        
        # Group metrics by prefix for better organization
        grouped = {}
        for key, value in sorted(all_metrics.items()):
            prefix = key.split('/')[0] if '/' in key else 'other'
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append((key, value))
        
        # Write grouped metrics
        for prefix in sorted(grouped.keys()):
            lines.append(f"{prefix.upper()}")
            lines.append("-" * 40)
            for key, value in grouped[prefix]:
                # Clean the key name (remove prefix)
                clean_key = key.split('/', 1)[1] if '/' in key else key
                # Format value properly
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        if abs(value) < 0.001 and value != 0:
                            value_str = f"{value:.6f}"
                        elif abs(value) < 0.01:
                            value_str = f"{value:.4f}"
                        elif abs(value) < 1:
                            value_str = f"{value:.3f}"
                        elif abs(value) < 1000:
                            value_str = f"{value:.2f}"
                        else:
                            value_str = f"{value:.0f}"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                
                lines.append(f"  {clean_key:<25} : {value_str:>10}")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("")
        
        # Write to log using standard logger - this will appear cleanly in the log file
        for line in lines:
            logger.info(line)