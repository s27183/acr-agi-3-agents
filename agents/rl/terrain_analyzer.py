"""
Terrain Analyzer for ARC-AGI-3 Games

This module provides terrain analysis capabilities to identify walkable floors
and wall areas based on color frequency analysis.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """Analyzes game grids to identify terrain types based on color frequencies."""
    
    def __init__(self):
        self.last_analysis = None
        
    def analyze_terrain(self, grid: np.ndarray, player_position: Optional[Tuple[int, int]] = None, 
                       player_shape: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Analyze grid to identify walkable floor and wall areas.
        
        Uses player position context to determine which of the two most frequent colors
        represents the walkable area vs walls, since the player starts on walkable terrain.
        
        Args:
            grid: 64x64 numpy array representing the game grid
            player_position: Optional (row, col) of player's top-left corner for context
            player_shape: Optional list of relative offsets for player shape
            
        Returns:
            Dict containing:
            - floor_color: Walkable area color (determined by player context)
            - wall_color: Wall/obstacle color
            - floor_mask: Boolean mask of walkable areas
            - wall_mask: Boolean mask of wall/obstacle areas
            - color_frequencies: Complete frequency distribution
            - analysis_method: 'player_context' or 'frequency_only'
        """
        try:
            # Get color frequency distribution
            unique_colors, counts = np.unique(grid, return_counts=True)
            
            # Sort by frequency (descending)
            freq_sorted_indices = np.argsort(counts)[::-1]
            sorted_colors = unique_colors[freq_sorted_indices]
            sorted_counts = counts[freq_sorted_indices]
            
            # Determine floor and wall colors using player context if available
            if player_position is not None and len(sorted_colors) >= 2:
                floor_color, wall_color, analysis_method = self._determine_walkable_color_by_player_context(
                    grid, player_position, player_shape, sorted_colors[:2]
                )
            else:
                # Fallback to frequency-based approach
                floor_color = sorted_colors[0]  # Highest frequency
                wall_color = sorted_colors[1] if len(sorted_colors) > 1 else floor_color
                analysis_method = 'frequency_only'
            
            # Create masks
            floor_mask = grid == floor_color
            wall_mask = grid == wall_color
            
            # Create frequency dictionary
            color_frequencies = dict(zip(unique_colors, counts))
            
            # Calculate coverage percentages
            total_pixels = grid.size
            floor_coverage = np.sum(floor_mask) / total_pixels * 100
            wall_coverage = np.sum(wall_mask) / total_pixels * 100
            
            analysis = {
                'floor_color': int(floor_color),
                'wall_color': int(wall_color),
                'floor_mask': floor_mask,
                'wall_mask': wall_mask,
                'color_frequencies': {int(k): int(v) for k, v in color_frequencies.items()},
                'floor_coverage_percent': floor_coverage,
                'wall_coverage_percent': wall_coverage,
                'total_colors': len(unique_colors),
                'grid_shape': grid.shape,
                'analysis_method': analysis_method
            }
            
            self.last_analysis = analysis
            
            logger.debug(f"Terrain analysis complete: "
                        f"floor_color={floor_color} ({floor_coverage:.1f}%), "
                        f"wall_color={wall_color} ({wall_coverage:.1f}%), "
                        f"total_colors={len(unique_colors)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze terrain: {e}")
            raise
    
    def _determine_walkable_color_by_player_context(self, grid: np.ndarray, 
                                                   player_position: Tuple[int, int],
                                                   player_shape: Optional[List[Tuple[int, int]]],
                                                   candidate_colors: np.ndarray) -> Tuple[int, int, str]:
        """
        Determine which of the candidate colors represents walkable area by analyzing
        the colors surrounding the player's initial position.
        
        Args:
            grid: Game grid
            player_position: Player's top-left corner position (row, col)
            player_shape: Player shape offsets (if None, uses default surrounding area)
            candidate_colors: Array of the 2 most frequent colors [color1, color2]
            
        Returns:
            Tuple of (floor_color, wall_color, analysis_method)
        """
        try:
            player_row, player_col = player_position
            color1, color2 = candidate_colors[0], candidate_colors[1]
            
            # Define surrounding area to analyze
            if player_shape is not None:
                # Use a buffer around the player shape
                surrounding_positions = self._get_player_surrounding_area(
                    player_position, player_shape, grid.shape, buffer_size=2
                )
            else:
                # Default: analyze a 5x5 area around player center
                surrounding_positions = self._get_default_surrounding_area(
                    player_position, grid.shape, area_size=5
                )
            
            # Count occurrences of each candidate color in surrounding area
            color1_count = 0
            color2_count = 0
            total_valid_positions = 0
            
            for row, col in surrounding_positions:
                if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                    cell_color = grid[row, col]
                    total_valid_positions += 1
                    
                    if cell_color == color1:
                        color1_count += 1
                    elif cell_color == color2:
                        color2_count += 1
            
            # Calculate percentages
            if total_valid_positions > 0:
                color1_percentage = (color1_count / total_valid_positions) * 100
                color2_percentage = (color2_count / total_valid_positions) * 100
            else:
                # Fallback to frequency-based if no valid surrounding positions
                logger.warning("No valid surrounding positions found, falling back to frequency-based analysis")
                return int(color1), int(color2), 'frequency_fallback'
            
            # The color with higher percentage around player is the walkable area
            if color1_percentage > color2_percentage:
                floor_color, wall_color = color1, color2
                dominant_percentage = color1_percentage
            else:
                floor_color, wall_color = color2, color1
                dominant_percentage = color2_percentage
            
            logger.info(f"Player context analysis: "
                       f"color_{floor_color} ({dominant_percentage:.1f}%) = walkable, "
                       f"color_{wall_color} = wall "
                       f"(analyzed {total_valid_positions} surrounding positions)")
            
            return int(floor_color), int(wall_color), 'player_context'
            
        except Exception as e:
            logger.error(f"Error in player context analysis: {e}")
            # Fallback to frequency-based
            return int(candidate_colors[0]), int(candidate_colors[1]), 'player_context_error'
    
    def _get_player_surrounding_area(self, player_position: Tuple[int, int], 
                                    player_shape: List[Tuple[int, int]], 
                                    grid_shape: Tuple[int, int],
                                    buffer_size: int = 2) -> List[Tuple[int, int]]:
        """
        Get positions surrounding the player shape with a buffer.
        
        Args:
            player_position: Player's top-left corner (row, col)
            player_shape: List of relative offsets defining player shape
            grid_shape: Shape of the grid (rows, cols)
            buffer_size: Number of cells to extend around player shape
            
        Returns:
            List of (row, col) positions surrounding the player
        """
        player_row, player_col = player_position
        
        # Find bounding box of player shape
        min_rel_row = min(offset[0] for offset in player_shape)
        max_rel_row = max(offset[0] for offset in player_shape)
        min_rel_col = min(offset[1] for offset in player_shape)
        max_rel_col = max(offset[1] for offset in player_shape)
        
        # Expand bounding box by buffer size
        start_row = player_row + min_rel_row - buffer_size
        end_row = player_row + max_rel_row + buffer_size + 1
        start_col = player_col + min_rel_col - buffer_size
        end_col = player_col + max_rel_col + buffer_size + 1
        
        # Collect all positions in expanded area, excluding player shape itself
        surrounding_positions = []
        player_absolute_positions = {
            (player_row + rel_row, player_col + rel_col) 
            for rel_row, rel_col in player_shape
        }
        
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if (row, col) not in player_absolute_positions:
                    surrounding_positions.append((row, col))
        
        return surrounding_positions
    
    def _get_default_surrounding_area(self, player_position: Tuple[int, int], 
                                     grid_shape: Tuple[int, int],
                                     area_size: int = 5) -> List[Tuple[int, int]]:
        """
        Get a default square area surrounding the player position.
        
        Args:
            player_position: Player's top-left corner (row, col)
            grid_shape: Shape of the grid (rows, cols)
            area_size: Size of the square area (odd number recommended)
            
        Returns:
            List of (row, col) positions surrounding the player
        """
        player_row, player_col = player_position
        
        # Create square area centered on player position
        half_size = area_size // 2
        start_row = player_row - half_size
        end_row = player_row + half_size + 1
        start_col = player_col - half_size
        end_col = player_col + half_size + 1
        
        surrounding_positions = []
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                # Exclude the player position itself
                if (row, col) != player_position:
                    surrounding_positions.append((row, col))
        
        return surrounding_positions
    
    def check_collision(self, position: Tuple[int, int], 
                       player_shape: List[Tuple[int, int]], 
                       wall_mask: np.ndarray) -> bool:
        """
        Check if player would collide with walls at given position.
        
        Args:
            position: Top-left corner of player (row, col)
            player_shape: List of relative pixel offsets from top-left
            wall_mask: Boolean mask where True = wall/obstacle
            
        Returns:
            True if collision detected, False if movement is valid
        """
        try:
            row, col = position
            
            for rel_row, rel_col in player_shape:
                abs_row = row + rel_row
                abs_col = col + rel_col
                
                # Check grid boundaries
                if abs_row < 0 or abs_row >= wall_mask.shape[0] or abs_col < 0 or abs_col >= wall_mask.shape[1]:
                    return True
                    
                # Check wall collision
                if wall_mask[abs_row, abs_col]:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking collision at {position}: {e}")
            return True  # Assume collision on error for safety
    
    def get_collision_type(self, position: Tuple[int, int], 
                          player_shape: List[Tuple[int, int]], 
                          wall_mask: np.ndarray) -> str:
        """
        Determine the type of collision at a given position.
        
        Args:
            position: Top-left corner of player (row, col)
            player_shape: List of relative pixel offsets from top-left
            wall_mask: Boolean mask where True = wall/obstacle
            
        Returns:
            'none', 'boundary', 'wall', or 'mixed'
        """
        try:
            row, col = position
            boundary_collision = False
            wall_collision = False
            
            for rel_row, rel_col in player_shape:
                abs_row = row + rel_row
                abs_col = col + rel_col
                
                # Check grid boundaries
                if abs_row < 0 or abs_row >= wall_mask.shape[0] or abs_col < 0 or abs_col >= wall_mask.shape[1]:
                    boundary_collision = True
                    continue
                    
                # Check wall collision
                if wall_mask[abs_row, abs_col]:
                    wall_collision = True
            
            if boundary_collision and wall_collision:
                return 'mixed'
            elif boundary_collision:
                return 'boundary'
            elif wall_collision:
                return 'wall'
            else:
                return 'none'
                
        except Exception as e:
            logger.error(f"Error determining collision type at {position}: {e}")
            return 'unknown'
    
    def find_safe_positions(self, player_shape: List[Tuple[int, int]], 
                           wall_mask: np.ndarray, 
                           sample_positions: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """
        Find all positions where the player can be placed without collision.
        
        Args:
            player_shape: List of relative pixel offsets from top-left
            wall_mask: Boolean mask where True = wall/obstacle
            sample_positions: Optional list of positions to check (if None, checks all)
            
        Returns:
            List of safe (row, col) positions
        """
        safe_positions = []
        
        if sample_positions is None:
            # Check all possible positions
            for row in range(wall_mask.shape[0]):
                for col in range(wall_mask.shape[1]):
                    if not self.check_collision((row, col), player_shape, wall_mask):
                        safe_positions.append((row, col))
        else:
            # Check only specified positions
            for position in sample_positions:
                if not self.check_collision(position, player_shape, wall_mask):
                    safe_positions.append(position)
        
        logger.debug(f"Found {len(safe_positions)} safe positions out of {len(sample_positions) if sample_positions else wall_mask.size} checked")
        
        return safe_positions
    
    def get_walkable_area_size(self, player_shape: List[Tuple[int, int]], 
                              wall_mask: np.ndarray) -> int:
        """
        Calculate the total number of positions where the player can be placed.
        
        Args:
            player_shape: List of relative pixel offsets from top-left
            wall_mask: Boolean mask where True = wall/obstacle
            
        Returns:
            Number of walkable positions
        """
        safe_positions = self.find_safe_positions(player_shape, wall_mask)
        return len(safe_positions)
    
    def create_walkable_mask(self, player_shape: List[Tuple[int, int]], 
                            wall_mask: np.ndarray) -> np.ndarray:
        """
        Create a mask showing all positions where the player top-left can be placed.
        
        Args:
            player_shape: List of relative pixel offsets from top-left
            wall_mask: Boolean mask where True = wall/obstacle
            
        Returns:
            Boolean mask where True = player can be placed at this top-left position
        """
        walkable_mask = np.zeros_like(wall_mask, dtype=bool)
        
        for row in range(wall_mask.shape[0]):
            for col in range(wall_mask.shape[1]):
                if not self.check_collision((row, col), player_shape, wall_mask):
                    walkable_mask[row, col] = True
        
        return walkable_mask