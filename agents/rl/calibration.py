"""
Color Transition Player Tracker for ARC-AGI-3 Games

This module provides player detection and tracking based on the (2N + 1) color transition
pattern discovered in ARC games, where N is the player size in pixels.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from collections import deque
from agents.rl.terrain_analyzer import TerrainAnalyzer

from agents.structs import FrameData

logger = logging.getLogger(__name__)


class Calibration:
    """Player tracking based on color transition patterns."""

    def __init__(self):
        self.player_size = None          # N pixels (dynamic detection)
        self.energy_consumption = None   # e pixels (energy consumption)
        self.player_location = None      # Current player position
        self.calibrated = False
        self.hotspot_map = None
        self.wall_mask = None           # Wall constraint optimization
        self.play_area_mask = None
        self.terrain_analyzer = TerrainAnalyzer()  # For terrain analysis
        self.terrain_map = None          # Cached terrain analysis

    def calibrate_at_reset(self, swarm_ops) -> tuple[bool, FrameData|None]:
        """
        Systematic calibration following the 7-point specification:
        1. Try actions 1-4 sequentially until pixel changes occur
        2. Use 2*N+e formula to determine player size N and energy consumption e
        3. Extract exact player location using rectangular shape detection
        4. Store post-calibration state as training starting point

        Args:
            swarm_ops: SwarmOperations instance for executing calibration actions

        Returns:
            tuple: (success, frame_data) - bool and optional FrameData
        """
        try:
            # Get initial reset state
            reset_frame = swarm_ops.last_frame
            if reset_frame is None:
                logger.warning("No reset frame available for calibration")
                return False, None

            reset_grid = self._frame_to_grid(reset_frame)
            reset_score = reset_frame.score

            # Initialize wall constraints for optimization
            self._initialize_wall_constraints(reset_grid)

            logger.info(f"Starting systematic calibration with grid shape {reset_grid.shape}, score {reset_score}")

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
                    if self.play_area_mask is not None:
                        total_changes = total_changes & self.play_area_mask

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

            # Step 2 & 3: Analyze calibrated move using 2*N+e formula and extract player geometry
            success = self._analyze_calibrated_move(
                reset_grid, calibrated_grid, calibrated_action, reset_score, calibrated_frame.score
            )

            if success:
                self.calibrated = True
                self._initialize_hotspot_map(reset_grid.shape)

                # Store post-calibration state for training
                self.post_calibration_grid = calibrated_grid.copy()
                self.post_calibration_frame = calibrated_frame

                logger.info(f"Calibration successful! Player size N={self.player_size}, "
                          f"energy consumption e={self.energy_consumption}")
                logger.info(f"Player top-left position: {self.player_location['current_position']}")
                logger.info(f"Training will start from post-calibration state")
                return True, test_frame
            else:
                logger.warning("Failed to analyze calibrated move with 2*N+e formula")
                return False, test_frame

        except Exception as e:
            logger.error(f"Calibration failed with exception: {e}")
            return False, test_frame

    def _analyze_calibrated_move(self, reset_grid, calibrated_grid, action, reset_score, calibrated_score):
        """
        Analyze calibrated move using 2*N+e formula and extract player geometry.

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

            # Apply play area mask if available
            if self.play_area_mask is not None:
                change_matrix = change_matrix * self.play_area_mask

            # Find all non-zero cells (any change from 0)
            changed_cells = (change_matrix != 0)
            total_changes_count = np.sum(changed_cells)

            # Log detailed change analysis
            unique_changes = np.unique(change_matrix[changed_cells])
            logger.info(f"Grid subtraction found {total_changes_count} changed cells")
            logger.info(f"Change values found: {unique_changes}")
            logger.debug(f"Change matrix range: min={np.min(change_matrix)}, max={np.max(change_matrix)}")

            # Step 2: Apply 2*N+e formula using ARC-AGI-3 specifications
            # Expected values: N=64, e=1, total_changes=129 (2*64+1)
            if total_changes_count == 129:
                # Use known ARC-AGI-3 specifications
                N = 64  # Player size is 64x64
                e = 1   # Energy consumption is 1 pixel
                logger.info(f"Using ARC-AGI-3 standard calibration: N={N}, e={e}")
            else:
                # Fall back to spatial pattern analysis for non-standard cases
                logger.info(f"Non-standard change count ({total_changes_count}), using spatial analysis")

                # Find connected components for spatial analysis
                player_movement_cells, energy_bar_cells = self._separate_player_from_energy_changes(
                    changed_cells, action
                )

                # Validate 2*N formula for player movement
                if len(player_movement_cells) % 2 != 0:
                    logger.warning(f"Invalid 2*N: player movement cells({len(player_movement_cells)}) not divisible by 2")
                    return False

                N = len(player_movement_cells) // 2  # Player size from movement pattern
                e = len(energy_bar_cells)           # Energy consumption from energy bar changes

            if N <= 0:
                logger.warning(f"Invalid player size N={N}")
                return False

            self.player_size = N
            self.energy_consumption = e
            logger.info(f"Derived player size N={N}, energy consumption e={e} from {total_changes_count} total changes")

            # Step 3: Extract rectangular player shape and exact position
            success = self._extract_rectangular_player_geometry(
                reset_grid, calibrated_grid, changed_cells, action, N
            )

            return success

        except Exception as e:
            logger.error(f"Exception in calibrated move analysis: {e}")
            return False

    def _extract_rectangular_player_geometry(self, reset_grid, calibrated_grid, changes_mask, action, player_size_N):
        """
        Extract exact player geometry from the 2*N changed pixels.

        The changed pixels should form a rectangular shape representing:
        - N pixels where player was (old position)
        - N pixels where player moved to (new position)

        Args:
            reset_grid: Grid before movement
            calibrated_grid: Grid after movement
            changes_mask: Boolean mask of all changed pixels
            action: Movement direction (1=up, 2=down, 3=left, 4=right)
            player_size_N: Expected player size in pixels

        Returns:
            bool: True if successful geometry extraction
        """
        try:
            # Find all changed pixel coordinates
            changed_coords = np.where(changes_mask)
            changed_pixels = list(zip(changed_coords[0], changed_coords[1]))

            logger.debug(f"Analyzing {len(changed_pixels)} changed pixels for rectangular player geometry")

            # Try to identify the rectangular region containing the player movement
            # We expect to find a contiguous rectangular region of 2*N pixels
            bounding_box = self._find_player_bounding_box(changed_pixels, player_size_N)

            if bounding_box is None:
                logger.warning("Could not identify rectangular player bounding box")
                return False

            # Analyze player movement within the bounding box
            player_positions = self._identify_old_new_player_positions(
                reset_grid, calibrated_grid, bounding_box, action, player_size_N
            )

            if player_positions is None:
                logger.warning("Could not identify old and new player positions")
                return False

            old_pos, new_pos, player_shape = player_positions

            # Store exact player location and geometry
            self.player_location = {
                'current_position': new_pos,  # Top-left corner after calibrated move
                'previous_position': old_pos,  # Top-left corner at reset
                'size': player_size_N,
                'shape': player_shape  # Relative coordinates of all player pixels
            }

            logger.info(f"Extracted player geometry: {player_size_N} pixels, "
                       f"moved from {old_pos} to {new_pos}")

            return True

        except Exception as e:
            logger.error(f"Exception in rectangular geometry extraction: {e}")
            return False

    def _find_player_bounding_box(self, changed_pixels, expected_size):
        """Find the minimal bounding box that could contain the player movement."""
        if len(changed_pixels) < 2 * expected_size:
            return None

        # Calculate bounding box of all changes
        rows = [p[0] for p in changed_pixels]
        cols = [p[1] for p in changed_pixels]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        return {
            'min_row': min_row, 'max_row': max_row,
            'min_col': min_col, 'max_col': max_col,
            'height': max_row - min_row + 1,
            'width': max_col - min_col + 1
        }

    def _identify_old_new_player_positions(self, reset_grid, calibrated_grid, bbox, action, player_size):
        """
        Identify old and new player positions using action direction and color changes.

        Returns:
            tuple: (old_top_left, new_top_left, player_shape) or None if failed
        """
        try:
            # Action direction vectors
            action_vectors = {
                1: (-1, 0),  # Up
                2: (1, 0),   # Down
                3: (0, -1),  # Left
                4: (0, 1)    # Right
            }

            if action not in action_vectors:
                return None

            dr, dc = action_vectors[action]

            # For ARC-AGI-3 standard case, use simplified position detection
            if player_size == 64:
                # Use center of bounding box as approximate position
                center_row = (bbox['min_row'] + bbox['max_row']) // 2
                center_col = (bbox['min_col'] + bbox['max_col']) // 2

                old_pos = (center_row, center_col)
                new_pos = (center_row + dr, center_col + dc)

                # Create simplified shape (single pixel for now)
                player_shape = [(0, 0)]

                return old_pos, new_pos, player_shape

            # For non-standard cases, use detailed analysis
            # Find all pixels that changed within the bounding box
            changed_pixels_in_bbox = []
            for r in range(bbox['min_row'], bbox['max_row'] + 1):
                for c in range(bbox['min_col'], bbox['max_col'] + 1):
                    if reset_grid[r, c] != calibrated_grid[r, c]:
                        changed_pixels_in_bbox.append((r, c))

            # Separate into old player pixels (became background) and new player pixels (became player)
            old_player_pixels = []
            new_player_pixels = []

            for r, c in changed_pixels_in_bbox:
                old_color = reset_grid[r, c]
                new_color = calibrated_grid[r, c]

                # Heuristic: assume player has non-zero color, background is often 0
                if old_color != 0 and new_color == 0:
                    # Player pixel became background (old position)
                    old_player_pixels.append((r, c))
                elif old_color == 0 and new_color != 0:
                    # Background became player pixel (new position)
                    new_player_pixels.append((r, c))
                elif old_color != 0 and new_color != 0:
                    # Color changed but both non-zero - could be either
                    # Use action direction to disambiguate
                    if self._pixel_in_direction_from_center(r, c, bbox, dr, dc):
                        new_player_pixels.append((r, c))
                    else:
                        old_player_pixels.append((r, c))

            # If we don't have clear old/new separation, use center-based fallback
            if not old_player_pixels or not new_player_pixels:
                center_row = (bbox['min_row'] + bbox['max_row']) // 2
                center_col = (bbox['min_col'] + bbox['max_col']) // 2

                old_pos = (center_row, center_col)
                new_pos = (center_row + dr, center_col + dc)
                player_shape = [(0, 0)]

                return old_pos, new_pos, player_shape

            # Calculate top-left corners
            old_rows, old_cols = zip(*old_player_pixels)
            new_rows, new_cols = zip(*new_player_pixels)

            old_top_left = (min(old_rows), min(old_cols))
            new_top_left = (min(new_rows), min(new_cols))

            # Create relative shape from new player position
            player_shape = []
            for r, c in new_player_pixels:
                rel_r = r - new_top_left[0]
                rel_c = c - new_top_left[1]
                player_shape.append((rel_r, rel_c))

            return old_top_left, new_top_left, player_shape

        except Exception as e:
            logger.error(f"Exception identifying player positions: {e}")
            return None

    def _pixel_in_direction_from_center(self, r, c, bbox, dr, dc):
        """Check if pixel is in the movement direction from bounding box center."""
        center_r = (bbox['min_row'] + bbox['max_row']) / 2
        center_c = (bbox['min_col'] + bbox['max_col']) / 2

        if dr != 0:  # Vertical movement
            return (r - center_r) * dr > 0
        else:  # Horizontal movement
            return (c - center_c) * dc > 0

    def _initialize_wall_constraints(self, grid):
        """Initialize wall and play area masks for optimization."""
        height, width = grid.shape
        wall_color = grid[0, 0]  # Assume corner is wall color

        # Create wall mask (edges + wall color pixels)
        wall_mask = np.zeros((height, width), dtype=bool)
        wall_mask[0, :] = True   # Top edge
        wall_mask[-1, :] = True  # Bottom edge
        wall_mask[:, 0] = True   # Left edge
        wall_mask[:, -1] = True  # Right edge
        wall_mask |= (grid == wall_color)

        self.wall_mask = wall_mask
        self.play_area_mask = ~wall_mask

    def _initialize_hotspot_map(self, grid_shape):
        """Initialize hotspot attention map."""
        self.hotspot_map = np.zeros(grid_shape, dtype=np.float32)

    def separate_changes(self, old_grid, new_grid, action_taken):
        """
        Separate player movement from environment changes using exact tracking.

        Following the 7-point specification:
        - No pixel change = wall collision, don't update location
        - Pixel changes = movement occurred, decompose into player (2N) + environment
        """
        if not self.calibrated:
            # Fallback: treat all changes as environment changes
            total_changes = (old_grid != new_grid)
            if self.play_area_mask is not None:
                total_changes = total_changes & self.play_area_mask

            return {
                'total_pixels_changed': np.sum(total_changes),
                'player_movement_pixels': 0,
                'environment_interaction_pixels': np.sum(total_changes),
                'environment_change_mask': total_changes,
                'player_movement_mask': np.zeros_like(total_changes),
                'calibrated': False,
                'wall_collision': False
            }

        # Calculate total changes (focus on play area)
        total_changes = (old_grid != new_grid)
        if self.play_area_mask is not None:
            total_changes = total_changes & self.play_area_mask

        changes_count = np.sum(total_changes)

        # Step 5: Wall collision detection via zero pixel changes
        wall_collision = (changes_count == 0)

        if wall_collision:
            logger.debug(f"Wall collision detected for action {action_taken} - no pixel changes")
            # Don't update player location
            return {
                'total_pixels_changed': 0,
                'player_movement_pixels': 0,
                'environment_interaction_pixels': 0,
                'environment_change_mask': np.zeros_like(total_changes),
                'player_movement_mask': np.zeros_like(total_changes),
                'calibrated': True,
                'wall_collision': True
            }

        # Step 6: Decompose changes into player movement + environmental changes
        player_movement_pixels = 0
        player_movement_mask = np.zeros_like(total_changes)

        if action_taken in [1, 2, 3, 4] and self.player_size is not None:
            # Expected player movement: 2*N pixels
            expected_player_pixels = 2 * self.player_size

            # Try to identify actual player movement pixels
            player_movement_mask = self._identify_player_movement_pixels(
                old_grid, new_grid, total_changes, action_taken
            )
            player_movement_pixels = np.sum(player_movement_mask)

            # Update player location since movement occurred
            self._update_player_location_exact(action_taken)

            logger.debug(f"Action {action_taken}: detected {player_movement_pixels} player pixels "
                        f"(expected {expected_player_pixels}), total changes: {changes_count}")

        # Environment changes = total changes - player movement
        # This includes energy consumption pixels (e) and other environmental interactions
        environment_changes = total_changes & ~player_movement_mask
        environment_pixels = np.sum(environment_changes)

        return {
            'total_pixels_changed': changes_count,
            'player_movement_pixels': player_movement_pixels,
            'environment_interaction_pixels': environment_pixels,
            'environment_change_mask': environment_changes,
            'player_movement_mask': player_movement_mask,
            'calibrated': True,
            'wall_collision': False
        }

    def _identify_player_movement_pixels(self, old_grid, new_grid, total_changes, action):
        """
        Identify which pixels correspond to player movement vs environmental changes.

        Uses the calibrated player location and action direction to predict
        where the player movement should occur.
        """
        if self.player_location is None:
            return np.zeros_like(total_changes)

        try:
            # Get current player position and shape
            current_pos = self.player_location['current_position']
            player_shape = self.player_location['shape']

            # Calculate where player should move based on action
            action_vectors = {
                1: (-1, 0),  # Up
                2: (1, 0),   # Down
                3: (0, -1),  # Left
                4: (0, 1)    # Right
            }

            if action not in action_vectors:
                return np.zeros_like(total_changes)

            dr, dc = action_vectors[action]
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)

            # Create masks for old and new player positions
            player_movement_mask = np.zeros_like(total_changes)

            # Mark old position pixels (where player was)
            for rel_r, rel_c in player_shape:
                old_r = current_pos[0] + rel_r
                old_c = current_pos[1] + rel_c
                if (0 <= old_r < total_changes.shape[0] and
                    0 <= old_c < total_changes.shape[1] and
                    total_changes[old_r, old_c]):
                    player_movement_mask[old_r, old_c] = True

            # Mark new position pixels (where player moved to)
            for rel_r, rel_c in player_shape:
                new_r = new_pos[0] + rel_r
                new_c = new_pos[1] + rel_c
                if (0 <= new_r < total_changes.shape[0] and
                    0 <= new_c < total_changes.shape[1] and
                    total_changes[new_r, new_c]):
                    player_movement_mask[new_r, new_c] = True

            return player_movement_mask

        except Exception as e:
            logger.debug(f"Exception in player movement identification: {e}")
            return np.zeros_like(total_changes)

    def _update_player_location_exact(self, action):
        """
        Update player location using exact position tracking.

        Since we know the exact player shape and action direction,
        we can update the position deterministically.
        """
        if self.player_location is None:
            return

        action_vectors = {
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1)    # Right
        }

        if action not in action_vectors:
            return

        dr, dc = action_vectors[action]
        current_pos = self.player_location['current_position']
        new_pos = (current_pos[0] + dr, current_pos[1] + dc)

        # Update location
        self.player_location = {
            'current_position': new_pos,
            'previous_position': current_pos,
            'size': self.player_location['size'],
            'shape': self.player_location['shape']
        }

        logger.debug(f"Player moved from {current_pos} to {new_pos} via action {action}")

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
            'energy_consumption': self.energy_consumption,
            'player_location': self.player_location
        }

    def reset(self):
        """Reset tracker state for new episode."""
        self.player_size = None
        self.energy_consumption = None
        self.player_location = None
        self.calibrated = False
        self.hotspot_map = None
        self.wall_mask = None
        self.play_area_mask = None
        self.terrain_map = None

    def _label_connected_components(self, binary_mask):
        """
        Simple connected component labeling using flood fill.
        Returns labeled array and number of components.
        """
        labels = np.zeros_like(binary_mask, dtype=np.int32)
        label_count = 0

        rows, cols = np.where(binary_mask)
        visited = np.zeros_like(binary_mask, dtype=bool)

        for r, c in zip(rows, cols):
            if not visited[r, c]:
                label_count += 1
                # Flood fill this component
                queue = deque([(r, c)])

                while queue:
                    cr, cc = queue.popleft()
                    if visited[cr, cc]:
                        continue

                    visited[cr, cc] = True
                    labels[cr, cc] = label_count

                    # Check 4-connected neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < binary_mask.shape[0] and
                            0 <= nc < binary_mask.shape[1] and
                            binary_mask[nr, nc] and
                            not visited[nr, nc]):
                            queue.append((nr, nc))

        return labels, label_count

    def _separate_player_from_energy_changes(self, changed_cells, action):
        """
        Separate player movement from energy consumption using adjacency patterns.

        Args:
            changed_cells: Boolean mask of all changed cells
            action: Movement action (1=up, 2=down, 3=left, 4=right)

        Returns:
            tuple: (player_movement_cells, energy_bar_cells) as lists of (row, col) coordinates
        """
        try:
            # Find connected components among changed cells
            labeled_components, num_components = self._label_connected_components(changed_cells)

            logger.debug(f"Found {num_components} connected components in changed cells")

            # Analyze each component to determine if it's player movement or energy bar
            player_movement_cells = []
            energy_bar_cells = []

            action_vectors = {
                1: (-1, 0),  # Up
                2: (1, 0),   # Down
                3: (0, -1),  # Left
                4: (0, 1)    # Right
            }

            dr, dc = action_vectors.get(action, (0, 0))

            for component_id in range(1, num_components + 1):
                component_mask = (labeled_components == component_id)
                component_coords = np.where(component_mask)
                component_size = len(component_coords[0])

                # Get component cells as list of coordinates
                component_cells = list(zip(component_coords[0], component_coords[1]))

                # Analyze component characteristics
                component_info = self._analyze_component_characteristics(
                    component_cells, action, dr, dc
                )

                logger.debug(f"Component {component_id}: size={component_size}, "
                           f"type={component_info['type']}, "
                           f"bbox={component_info['bbox']}")

                # Classify component as player movement or energy bar
                if component_info['type'] == 'player_movement':
                    player_movement_cells.extend(component_cells)
                elif component_info['type'] == 'energy_bar':
                    energy_bar_cells.extend(component_cells)
                else:
                    # Unknown type - for now, assume it's energy/environment
                    energy_bar_cells.extend(component_cells)
                    logger.debug(f"Component {component_id} classified as energy/environment (unknown type)")

            logger.info(f"Separated changes: {len(player_movement_cells)} player movement cells, "
                       f"{len(energy_bar_cells)} energy bar cells")

            return player_movement_cells, energy_bar_cells

        except Exception as e:
            logger.error(f"Exception in separating player from energy changes: {e}")
            # Fallback: assume all changes are energy/environment
            changed_coords = np.where(changed_cells)
            all_cells = list(zip(changed_coords[0], changed_coords[1]))
            return [], all_cells

    def _analyze_component_characteristics(self, component_cells, action, dr, dc):
        """
        Analyze a connected component to determine if it's player movement or energy bar.

        Args:
            component_cells: List of (row, col) coordinates in the component
            action: Movement action
            dr, dc: Action direction vectors

        Returns:
            dict: Component analysis with 'type', 'bbox', 'shape_ratio', etc.
        """
        if not component_cells:
            return {'type': 'unknown', 'bbox': None}

        # Calculate bounding box
        rows = [cell[0] for cell in component_cells]
        cols = [cell[1] for cell in component_cells]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        bbox = {
            'min_row': min_row, 'max_row': max_row,
            'min_col': min_col, 'max_col': max_col,
            'height': max_row - min_row + 1,
            'width': max_col - min_col + 1
        }

        # Calculate shape characteristics
        component_size = len(component_cells)
        bbox_area = bbox['height'] * bbox['width']
        fill_ratio = component_size / bbox_area if bbox_area > 0 else 0
        aspect_ratio = bbox['width'] / bbox['height'] if bbox['height'] > 0 else 1

        # Determine component type based on characteristics
        component_type = self._classify_component_type(
            component_size, bbox, fill_ratio, aspect_ratio, action, dr, dc
        )

        return {
            'type': component_type,
            'bbox': bbox,
            'size': component_size,
            'fill_ratio': fill_ratio,
            'aspect_ratio': aspect_ratio
        }

    def _classify_component_type(self, size, bbox, fill_ratio, aspect_ratio, action, dr, dc):
        """
        Classify a component as player movement or energy bar based on characteristics.

        Heuristics:
        - Player movement: Usually compact, moderate size, aligned with action direction
        - Energy bar: Often linear/elongated, may be positioned at edges
        """

        # Heuristic 1: Size-based classification
        if size >= 20 and size <= 200:  # Reasonable player size range
            # Check if component is aligned with movement direction
            if abs(dr) > 0:  # Vertical movement
                if bbox['height'] >= bbox['width']:  # Taller than wide
                    return 'player_movement'
            elif abs(dc) > 0:  # Horizontal movement
                if bbox['width'] >= bbox['height']:  # Wider than tall
                    return 'player_movement'

        # Heuristic 2: Energy bar characteristics
        # Energy bars are often small, linear, or positioned at edges
        if (size < 20 or  # Small components likely energy
            aspect_ratio > 5 or aspect_ratio < 0.2 or  # Very elongated
            bbox['min_row'] == 0 or bbox['max_row'] == 63 or  # At top/bottom edge
            bbox['min_col'] == 0 or bbox['max_col'] == 63):   # At left/right edge
            return 'energy_bar'

        # Heuristic 3: Fill ratio
        if fill_ratio < 0.3:  # Very sparse components likely energy/UI
            return 'energy_bar'
        elif fill_ratio > 0.7 and 20 <= size <= 100:  # Dense, moderate size likely player
            return 'player_movement'

        # Default: classify as energy/environment if uncertain
        return 'energy_bar'