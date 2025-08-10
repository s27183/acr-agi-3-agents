"""
Simplified Color Transition Player Tracker for ARC-AGI-3 Games

This module provides simplified player size and energy consumption detection
using the 2*N+e formula discovered in ARC games.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from agents.structs import FrameData

logger = logging.getLogger(__name__)


class Calibration:
    """Player tracking based on color transition patterns."""
    
    def __init__(self):
        self.player_size = None          # N pixels (dynamic detection)
        self.energy_consumption = None   # e pixels (energy consumption)
        self.calibrated = False
        
    def calibrate_at_reset(self, swarm_ops) -> tuple[bool, FrameData|None]:
        """
        Simplified calibration focusing only on player size and energy consumption:
        1. Try actions 1-4 sequentially until pixel changes occur
        2. Use 2*N+e formula to determine player size N and energy consumption e
        
        Args:
            swarm_ops: SwarmOperations instance for executing calibration actions
            
        Returns:
            bool: True if calibration successful, False otherwise
            FrameData: Frame data from last successful calibration step
        """
        try:
            # Get initial reset state
            reset_frame = swarm_ops.last_frame
            if reset_frame is None:
                logger.warning("No reset frame available for calibration")
                return False
                
            reset_grid = self._frame_to_grid(reset_frame)
            reset_score = reset_frame.score
            
            logger.info(f"Starting simplified calibration with grid shape {reset_grid.shape}, score {reset_score}")
            
            # Step 1: Try actions 1-4 sequentially until finding pixel changes
            calibrated_action = None
            calibrated_frame = None
            calibrated_grid = None
            
            test_frame = None
            for action_id in [1, 2, 3, 4]:  # Up, Down, Left, Right
                try:
                    logger.debug(f"Testing calibration action {action_id}")
                    
                    # Execute test action
                    test_frame = swarm_ops.execute_action(action_id)
                    
                    if test_frame is None:
                        logger.debug(f"Action {action_id} failed during calibration")
                        continue
                        
                    test_grid = self._frame_to_grid(test_frame)
                    test_score = test_frame.score
                    
                    # Check for pixel changes
                    total_changes = (reset_grid != test_grid)
                    changes_count = np.sum(total_changes)
                    
                    logger.debug(f"Action {action_id}: {changes_count} pixels changed, score {reset_score}->{test_score}")
                    
                    # Check if this is a valid calibrated move (pixel changes occurred)
                    if changes_count > 0:
                        calibrated_action = action_id
                        calibrated_frame = test_frame
                        calibrated_grid = test_grid
                        logger.info(f"Found calibrated move: Action {action_id} with {changes_count} pixel changes")
                        break
                        
                except Exception as e:
                    logger.debug(f"Exception during calibration action {action_id}: {e}")
                    continue
                    
            if calibrated_action is None:
                logger.warning("Calibration failed - no action produced pixel changes")
                return False, test_frame
            
            # Step 2: Analyze calibrated move using 2*N+e formula
            success = self._analyze_calibrated_move(
                reset_grid, calibrated_grid, calibrated_action, reset_score, calibrated_frame.score
            )
            
            if success:
                self.calibrated = True
                logger.info(f"Calibration successful! Player size N={self.player_size}, "
                          f"energy consumption e={self.energy_consumption}")
                return True, test_frame
            else:
                logger.warning("Failed to analyze calibrated move with 2*N+e formula")
                return False, test_frame
                
        except Exception as e:
            logger.error(f"Calibration failed with exception: {e}")
            return False, test_frame
            
    def _analyze_calibrated_move(self, reset_grid, calibrated_grid, action, reset_score, calibrated_score):
        """
        Analyze calibrated move using 2*N+e formula to extract player size and energy consumption.
        
        Args:
            reset_grid: Grid state at reset
            calibrated_grid: Grid state after calibrated move
            action: The calibrated action (1=up, 2=down, 3=left, 4=right)
            reset_score: Score at reset
            calibrated_score: Score after calibrated move
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            # Step 1: Simple grid subtraction to find all changes
            change_matrix = calibrated_grid - reset_grid
            
            # Find all non-zero cells (any change from 0)
            changed_cells = (change_matrix != 0)
            total_changes_count = np.sum(changed_cells)
            
            logger.info(f"Grid subtraction found {total_changes_count} changed cells")
            
            # Step 2: Apply 2*N+e formula using simple heuristics
            # For simplicity, assume roughly 2/3 of changes are player movement, 1/3 is energy
            if total_changes_count < 3:
                logger.warning(f"Too few changes ({total_changes_count}) to calibrate reliably")
                return False
            
            # Simple heuristic: assume energy consumption is small (1-5 pixels)
            # Most changes should be player movement (2*N)
            estimated_energy = min(5, max(1, total_changes_count // 10))  # 1-5 pixels
            estimated_player_movement = total_changes_count - estimated_energy
            
            # Validate that player movement is even (2*N)
            if estimated_player_movement % 2 != 0:
                estimated_player_movement -= 1  # Round down to nearest even
                estimated_energy += 1
            
            N = estimated_player_movement // 2  # Player size
            e = estimated_energy               # Energy consumption
            
            if N <= 0:
                logger.warning(f"Invalid player size N={N}")
                return False
            
            self.player_size = N
            self.energy_consumption = e
            logger.info(f"Derived player size N={N}, energy consumption e={e} from {total_changes_count} total changes")
            
            return True
            
        except Exception as e:
            logger.error(f"Exception in calibrated move analysis: {e}")
            return False

            
    def separate_changes(self, old_grid, new_grid, action_taken):
        """
        Simplified change analysis for basic environment interaction tracking.
        
        Args:
            old_grid: Previous grid state
            new_grid: Current grid state  
            action_taken: Action that was taken
            
        Returns:
            Dict with basic change statistics
        """
        if not self.calibrated:
            # Fallback: treat all changes as environment changes
            total_changes = (old_grid != new_grid)
            
            return {
                'total_pixels_changed': np.sum(total_changes),
                'player_movement_pixels': 0,
                'environment_interaction_pixels': np.sum(total_changes),
                'environment_change_mask': total_changes,
                'player_movement_mask': np.zeros_like(total_changes),
                'calibrated': False,
                'wall_collision': False
            }
            
        # Calculate total changes
        total_changes = (old_grid != new_grid)
        changes_count = np.sum(total_changes)
        
        # Wall collision detection via zero pixel changes
        wall_collision = (changes_count == 0)
        
        if wall_collision:
            logger.debug(f"Wall collision detected for action {action_taken} - no pixel changes")
            return {
                'total_pixels_changed': 0,
                'player_movement_pixels': 0,
                'environment_interaction_pixels': 0,
                'environment_change_mask': np.zeros_like(total_changes),
                'player_movement_mask': np.zeros_like(total_changes),
                'calibrated': True,
                'wall_collision': True
            }
        
        # Estimate player movement and environment changes using calibrated values
        if action_taken in [1, 2, 3, 4] and self.player_size is not None:
            expected_player_pixels = 2 * self.player_size
            expected_energy_pixels = self.energy_consumption
            expected_baseline = expected_player_pixels + expected_energy_pixels
            
            # Environment changes = total - baseline expected changes
            environment_pixels = max(0, changes_count - expected_baseline)
            player_movement_pixels = min(expected_player_pixels, changes_count)
        else:
            # Unknown action or not calibrated - treat all as environment
            environment_pixels = changes_count
            player_movement_pixels = 0
        
        return {
            'total_pixels_changed': changes_count,
            'player_movement_pixels': player_movement_pixels,
            'environment_interaction_pixels': environment_pixels,
            'environment_change_mask': total_changes,
            'player_movement_mask': np.zeros_like(total_changes),  # Simplified - no precise tracking
            'calibrated': True,
            'wall_collision': False
        }
        
        
    def _frame_to_grid(self, frame_data) -> np.ndarray:
        """Convert frame data to 64x64 grid."""
        if frame_data is None or not frame_data.frame:
            return np.zeros((64, 64), dtype=np.int32)
            
        grid_data = frame_data.frame
        
        # Handle different grid formats
        if isinstance(grid_data, list):
            grid = np.array(grid_data, dtype=np.int32)
            if len(grid.shape) == 3:
                grid = grid[0]  # Take first frame
        else:
            grid = np.array(grid_data, dtype=np.int32)
            
        # Ensure 64x64
        if grid.shape != (64, 64):
            padded_grid = np.zeros((64, 64), dtype=np.int32)
            min_h = min(grid.shape[0], 64)
            min_w = min(grid.shape[1] if len(grid.shape) > 1 else 1, 64)
            
            if len(grid.shape) == 1 and grid.shape[0] == 4096:
                grid = grid.reshape(64, 64)
            elif len(grid.shape) >= 2:
                padded_grid[:min_h, :min_w] = grid[:min_h, :min_w]
                grid = padded_grid
            else:
                return padded_grid
                
        return grid
        
    
    
    def create_stacked_observation(self, current_grid, grid_history=None, temporal_depth=10):
        """
        Create fixed 10-channel temporal observation by stacking historical grids.
        
        This follows AlphaGo's approach of using temporal history for better
        causal understanding and action-consequence learning.
        
        Args:
            current_grid: Current grid state (64, 64)
            grid_history: List of previous grids [t-1, t-2, t-3, ...] (most recent first)
            temporal_depth: Fixed to 10 channels (parameter kept for compatibility)
            
        Returns:
            Fixed 10-channel observation (10, 64, 64):
            - Channel 0: Current grid (t)
            - Channel 1: Previous grid (t-1) 
            - Channels 2-9: Grids at t-2 through t-9 (if available)
            
        Notes:
            - Missing historical grids are zero-padded
            - Always returns exactly 10 channels
            - Follows AlphaGo Zero's temporal stacking approach
        """
        channels = []
        temporal_depth = 10  # Force fixed temporal depth
        
        # Channel 0: Always current grid
        channels.append(current_grid.copy())
        
        # Additional channels: Historical grids (channels 1-9)
        for i in range(1, temporal_depth):
            if grid_history is not None and len(grid_history) >= i:
                # Use historical grid (grid_history[0] is t-1, grid_history[1] is t-2, etc.)
                historical_grid = grid_history[i-1].copy()
            else:
                # Zero padding for missing history (early in episode)
                historical_grid = np.zeros_like(current_grid)
            
            channels.append(historical_grid)
        
        # Stack channels: (temporal_depth, 64, 64)
        observation = np.stack(channels, axis=0)
        
        return observation
        
    def get_state_info(self):
        """Get current tracker state for debugging."""
        return {
            'calibrated': self.calibrated,
            'player_size': self.player_size,
            'energy_consumption': self.energy_consumption
        }
    
        
    def reset(self):
        """Reset tracker state for new episode."""
        self.player_size = None
        self.energy_consumption = None
        self.calibrated = False
