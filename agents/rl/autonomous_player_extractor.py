"""
Autonomous Player Coordinate Extractor for ARC-AGI-3 Games

This module provides fully autonomous player detection and coordinate extraction
from frame differences without requiring prior knowledge of player size (N),
energy consumption (e), or movement direction.

The algorithm identifies player movement by finding the largest pair of similar-sized
connected components that are spatially adjacent, representing old and new player positions.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """Represents a connected component in the difference grid."""
    coordinates: List[Tuple[int, int]]                # List of (row, col) coordinates
    size: int = None                                  # Number of cells in component
    centroid: Tuple[float, float] = None              # Center point (row, col)
    bounding_box: Dict[str, int] = None               # min_row, max_row, min_col, max_col
    
    def __post_init__(self):
        if not self.coordinates:
            raise ValueError("Component must have at least one coordinate")
        
        # Calculate derived properties if not provided
        rows = [coord[0] for coord in self.coordinates]
        cols = [coord[1] for coord in self.coordinates]
        
        if self.size is None:
            self.size = len(self.coordinates)
        if self.centroid is None:
            self.centroid = (np.mean(rows), np.mean(cols))
        if self.bounding_box is None:
            self.bounding_box = {
                'min_row': min(rows), 'max_row': max(rows),
                'min_col': min(cols), 'max_col': max(cols)
            }


@dataclass
class ComponentPair:
    """Represents a pair of components that could be old/new player positions."""
    comp1: Component
    comp2: Component
    similarity_score: float    # How similar in size (0-1, higher is better)
    proximity_score: float     # How spatially close (0-1, higher is better)
    dominance_score: float     # What fraction of total changes (0-1, higher is better)
    alignment_score: float     # How well aligned for single movement (0-1, higher is better)
    total_score: float         # Combined weighted score
    movement_vector: Tuple[float, float]  # Direction vector from comp1 to comp2


def extract_player_coordinates(frame_0: np.ndarray, frame_1: np.ndarray) -> Dict[str, Any]:
    """
    Extract player coordinates from two frames using autonomous pattern detection.
    
    Args:
        frame_0: Initial frame (64x64 grid)
        frame_1: Frame after player movement (64x64 grid)
    
    Returns:
        dict: {
            'player_coords': List of (row, col) coordinates occupied by player,
            'movement_direction': int (1=up, 2=down, 3=left, 4=right, 0=unknown),
            'player_size': int (number of cells occupied by player),
            'confidence': float (0-1, confidence in detection),
            'old_player_coords': List of (row, col) coordinates of previous position,
            'energy_consumption': int (estimated e value),
            'total_changes': int (total cells changed),
            'player_colors': List of color values (0-15) at current player coordinates,
            'old_player_colors': List of color values (0-15) at previous player coordinates,
            'walkable_area_color': int (0-15) - the walkable terrain color,
            'wall_area_color': int (0-15) - the wall/boundary color,
            'terrain_confidence': float (0-1) - confidence in terrain detection
        }
    """
    try:
        logger.debug("Starting autonomous player coordinate extraction")
        
        # Step 1: Calculate difference grid and validate movement occurred
        diff_grid = frame_1.astype(np.int32) - frame_0.astype(np.int32)
        changed_cells = (diff_grid != 0)
        total_changes = np.sum(changed_cells)
        
        if total_changes == 0:
            logger.info("No changes detected - no movement occurred")
            return _create_no_movement_result()
        
        logger.debug(f"Found {total_changes} changed cells")
        
        # Step 2: Find all connected components
        components = find_connected_components(diff_grid)
        logger.debug(f"Identified {len(components)} connected components")
        
        if len(components) < 2:
            logger.warning("Insufficient components for player pair detection")
            return _create_failed_result(total_changes)
        
        # Step 3: Find optimal player component pair
        best_pair = find_optimal_player_pair(components, total_changes)
        
        if best_pair is None:
            logger.warning("Could not identify valid player component pair")
            return _create_failed_result(total_changes)
        
        logger.info(f"Found player pair with confidence {best_pair.total_score:.3f}")
        
        # Step 4: Infer movement direction (needs diff_grid to determine old vs new)
        movement_direction = infer_movement_direction(best_pair, diff_grid)
        
        # Step 5: Select new player position and extract coordinates
        new_player_coords, old_player_coords = select_player_positions(best_pair, movement_direction, diff_grid)
        
        # Step 6: Extract player colors from frames
        player_colors = [int(frame_1[coord[0], coord[1]]) for coord in new_player_coords]
        old_player_colors = [int(frame_0[coord[0], coord[1]]) for coord in old_player_coords]
        
        # Step 7: Analyze terrain colors (walkable vs wall areas)
        terrain_result = _analyze_terrain_colors(frame_1, new_player_coords, player_colors)
        
        # Step 8: Calculate derived metrics
        player_size = len(new_player_coords)
        energy_consumption = total_changes - (2 * player_size)
        
        result = {
            'player_coords': new_player_coords,
            'movement_direction': movement_direction,
            'player_size': player_size,
            'confidence': best_pair.total_score,
            'old_player_coords': old_player_coords,
            'energy_consumption': max(0, energy_consumption),
            'total_changes': total_changes,
            'player_colors': player_colors,
            'old_player_colors': old_player_colors,
            'walkable_area_color': terrain_result['walkable_area_color'],
            'wall_area_color': terrain_result['wall_area_color'],
            'terrain_confidence': terrain_result['terrain_confidence'],
            'success': True
        }
        
        logger.info(f"Successfully extracted player coordinates: {player_size} cells, "
                   f"direction={movement_direction}, energy={energy_consumption}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception in autonomous player extraction: {e}")
        return _create_failed_result(0)


def find_connected_components(diff_grid: np.ndarray) -> List[Component]:
    """
    Find all connected components in the difference grid using flood fill.
    
    Separates positive and negative changes to handle adjacent player movements.
    
    Args:
        diff_grid: Difference grid (frame_1 - frame_0)
    
    Returns:
        List of Component objects representing connected regions
    """
    height, width = diff_grid.shape
    components = []
    
    # Process positive and negative changes separately to handle adjacent movements
    for sign in [1, -1]:  # Positive changes first, then negative
        visited = np.zeros((height, width), dtype=bool)
        
        # Find cells with the current sign
        if sign == 1:
            target_cells = np.where(diff_grid > 0)
        else:
            target_cells = np.where(diff_grid < 0)
        
        for row, col in zip(target_cells[0], target_cells[1]):
            if visited[row, col]:
                continue
            
            # Flood fill this component (same sign only)
            component_coords = []
            queue = deque([(row, col)])
            
            while queue:
                r, c = queue.popleft()
                
                if (r < 0 or r >= height or c < 0 or c >= width or 
                    visited[r, c]):
                    continue
                
                # Check if cell has same sign as target
                if sign == 1 and diff_grid[r, c] <= 0:
                    continue
                elif sign == -1 and diff_grid[r, c] >= 0:
                    continue
                
                visited[r, c] = True
                component_coords.append((r, c))
                
                # Add 4-connected neighbors (same sign constraint checked above)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((r + dr, c + dc))
            
            if component_coords:
                components.append(Component(coordinates=component_coords))
    
    # Sort components by size (largest first) for better processing
    components.sort(key=lambda c: c.size, reverse=True)
    
    logger.debug(f"Found components with sizes: {[c.size for c in components]}")
    
    return components


def find_optimal_player_pair(components: List[Component], total_changes: int) -> Optional[ComponentPair]:
    """
    Find the best pair of components representing old and new player positions.
    
    Args:
        components: List of all connected components
        total_changes: Total number of changed cells
    
    Returns:
        ComponentPair with highest confidence score, or None if no valid pair found
    """
    if len(components) < 2:
        return None
    
    best_pair = None
    best_score = 0.0
    
    # Evaluate all possible component pairs
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp1, comp2 = components[i], components[j]
            
            # Calculate scoring metrics
            similarity_score = calculate_size_similarity(comp1, comp2)
            proximity_score = calculate_spatial_proximity(comp1, comp2)
            dominance_score = calculate_dominance_ratio(comp1, comp2, total_changes)
            alignment_score = calculate_movement_alignment(comp1, comp2)
            
            # Weighted total score (can be tuned based on testing)
            total_score = (
                0.35 * similarity_score +    # Size similarity is very important
                0.25 * proximity_score +     # Spatial proximity matters
                0.25 * dominance_score +     # Should be major part of changes
                0.15 * alignment_score       # Movement alignment
            )
            
            if total_score > best_score:
                # Calculate movement vector (from comp1 to comp2)
                movement_vector = (
                    comp2.centroid[0] - comp1.centroid[0],
                    comp2.centroid[1] - comp1.centroid[1]
                )
                
                best_pair = ComponentPair(
                    comp1=comp1, comp2=comp2,
                    similarity_score=similarity_score,
                    proximity_score=proximity_score,
                    dominance_score=dominance_score,
                    alignment_score=alignment_score,
                    total_score=total_score,
                    movement_vector=movement_vector
                )
                best_score = total_score
    
    # Only return pairs with reasonable confidence
    if best_pair and best_pair.total_score >= 0.4:  # Minimum confidence threshold
        logger.debug(f"Best pair scores: similarity={best_pair.similarity_score:.3f}, "
                    f"proximity={best_pair.proximity_score:.3f}, "
                    f"dominance={best_pair.dominance_score:.3f}, "
                    f"alignment={best_pair.alignment_score:.3f}")
        return best_pair
    
    return None


def calculate_size_similarity(comp1: Component, comp2: Component) -> float:
    """Calculate how similar two components are in size (0-1, higher is better)."""
    size_diff = abs(comp1.size - comp2.size)
    max_size = max(comp1.size, comp2.size)
    
    if max_size == 0:
        return 1.0
    
    return 1.0 - (size_diff / max_size)


def calculate_spatial_proximity(comp1: Component, comp2: Component) -> float:
    """Calculate spatial proximity between components (0-1, higher is better)."""
    # Calculate distance between centroids
    centroid_distance = np.sqrt(
        (comp1.centroid[0] - comp2.centroid[0]) ** 2 +
        (comp1.centroid[1] - comp2.centroid[1]) ** 2
    )
    
    # Calculate bounding box overlap or separation
    bb1, bb2 = comp1.bounding_box, comp2.bounding_box
    
    # Check for bounding box overlap
    horizontal_overlap = (bb1['max_col'] >= bb2['min_col'] and bb2['max_col'] >= bb1['min_col'])
    vertical_overlap = (bb1['max_row'] >= bb2['min_row'] and bb2['max_row'] >= bb1['min_row'])
    
    if horizontal_overlap and vertical_overlap:
        # Overlapping bounding boxes - very close
        return 1.0
    
    # Calculate minimum distance between bounding boxes
    if horizontal_overlap:
        # Horizontally aligned, measure vertical distance
        if bb1['max_row'] < bb2['min_row']:
            bbox_distance = bb2['min_row'] - bb1['max_row']
        else:
            bbox_distance = bb1['min_row'] - bb2['max_row']
    elif vertical_overlap:
        # Vertically aligned, measure horizontal distance
        if bb1['max_col'] < bb2['min_col']:
            bbox_distance = bb2['min_col'] - bb1['max_col']
        else:
            bbox_distance = bb1['min_col'] - bb2['max_col']
    else:
        # No alignment, use centroid distance as approximation
        bbox_distance = centroid_distance
    
    # Convert distance to proximity score (closer = higher score)
    # Use exponential decay to favor very close components
    max_reasonable_distance = 10  # Grid cells
    proximity = np.exp(-bbox_distance / max_reasonable_distance)
    
    return min(1.0, proximity)


def calculate_dominance_ratio(comp1: Component, comp2: Component, total_changes: int) -> float:
    """Calculate what fraction of total changes these components represent."""
    combined_size = comp1.size + comp2.size
    
    if total_changes == 0:
        return 0.0
    
    ratio = combined_size / total_changes
    return min(1.0, ratio)  # Cap at 1.0


def calculate_movement_alignment(comp1: Component, comp2: Component) -> float:
    """Calculate how well the component pair aligns with single-direction movement."""
    # Calculate centroid displacement
    dr = comp2.centroid[0] - comp1.centroid[0]
    dc = comp2.centroid[1] - comp1.centroid[1]
    
    # Calculate the ratio of primary to secondary movement
    abs_dr, abs_dc = abs(dr), abs(dc)
    
    if abs_dr + abs_dc == 0:
        # No movement - perfect alignment for stationary case
        return 1.0
    
    # Calculate alignment score based on how much movement is in primary direction
    primary_movement = max(abs_dr, abs_dc)
    total_movement = abs_dr + abs_dc
    
    alignment = primary_movement / total_movement
    
    # Boost score for pure horizontal or vertical movement
    if abs_dr == 0 or abs_dc == 0:
        alignment = 1.0
    
    return alignment


def infer_movement_direction(pair: ComponentPair, diff_grid: np.ndarray) -> int:
    """
    Infer movement direction from component pair geometry.
    
    Args:
        pair: Component pair representing player movement
        diff_grid: Difference grid to identify positive/negative changes
    
    Returns:
        int: 1=up, 2=down, 3=left, 4=right, 0=unknown
    """
    # First identify which component is old (negative) and which is new (positive)
    coord1 = pair.comp1.coordinates[0] if pair.comp1.coordinates else (0, 0)
    coord2 = pair.comp2.coordinates[0] if pair.comp2.coordinates else (0, 0)
    
    value1 = diff_grid[coord1[0], coord1[1]]
    value2 = diff_grid[coord2[0], coord2[1]]
    
    # Determine old and new positions based on signs
    if value1 > 0 and value2 < 0:
        # comp1 is new (positive), comp2 is old (negative)
        new_centroid, old_centroid = pair.comp1.centroid, pair.comp2.centroid
    elif value1 < 0 and value2 > 0:
        # comp2 is new (positive), comp1 is old (negative)  
        new_centroid, old_centroid = pair.comp2.centroid, pair.comp1.centroid
    else:
        # Ambiguous case - use existing movement vector
        dr, dc = pair.movement_vector
        if abs(dr) > abs(dc):
            return 1 if dr < 0 else 2
        elif abs(dc) > abs(dr):
            return 3 if dc < 0 else 4
        else:
            return 0
    
    # Calculate movement vector from old to new position
    dr = new_centroid[0] - old_centroid[0]
    dc = new_centroid[1] - old_centroid[1]
    
    # Determine primary movement direction  
    if abs(dr) > abs(dc):
        # Vertical movement
        return 1 if dr < 0 else 2  # Up if negative, down if positive
    elif abs(dc) > abs(dr):
        # Horizontal movement
        return 3 if dc < 0 else 4  # Left if negative, right if positive
    else:
        # Diagonal or no clear direction
        return 0


def select_player_positions(pair: ComponentPair, movement_direction: int, diff_grid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Select which component is the new player position and which is the old.
    
    Uses the insight that positive changes in diff_grid represent new player positions,
    and negative changes represent old player positions.
    
    Args:
        pair: Component pair representing player movement
        movement_direction: Inferred movement direction
        diff_grid: Difference grid to check positive/negative values
    
    Returns:
        Tuple of (new_player_coords, old_player_coords)
    """
    # Check the sign of changes in each component
    coord1 = pair.comp1.coordinates[0] if pair.comp1.coordinates else (0, 0)
    coord2 = pair.comp2.coordinates[0] if pair.comp2.coordinates else (0, 0)
    
    value1 = diff_grid[coord1[0], coord1[1]]
    value2 = diff_grid[coord2[0], coord2[1]]
    
    # Positive values = new player position, negative values = old player position
    if value1 > 0 and value2 < 0:
        new_comp, old_comp = pair.comp1, pair.comp2
    elif value1 < 0 and value2 > 0:
        new_comp, old_comp = pair.comp2, pair.comp1
    else:
        # Fallback to movement direction logic if signs are ambiguous
        dr, dc = pair.movement_vector
        
        if movement_direction in [1, 2]:  # Vertical movement
            if movement_direction == 1:  # Up (negative dr)
                # New position has smaller row number
                new_comp = pair.comp1 if pair.comp1.centroid[0] < pair.comp2.centroid[0] else pair.comp2
            else:  # Down (positive dr)  
                # New position has larger row number
                new_comp = pair.comp1 if pair.comp1.centroid[0] > pair.comp2.centroid[0] else pair.comp2
        elif movement_direction in [3, 4]:  # Horizontal movement
            if movement_direction == 3:  # Left (negative dc)
                # New position has smaller column number
                new_comp = pair.comp1 if pair.comp1.centroid[1] < pair.comp2.centroid[1] else pair.comp2
            else:  # Right (positive dc)
                # New position has larger column number  
                new_comp = pair.comp1 if pair.comp1.centroid[1] > pair.comp2.centroid[1] else pair.comp2
        else:
            # Unknown direction - use larger component as new position (arbitrary choice)
            new_comp = pair.comp1 if pair.comp1.size >= pair.comp2.size else pair.comp2
        
        old_comp = pair.comp2 if new_comp == pair.comp1 else pair.comp1
    
    return new_comp.coordinates, old_comp.coordinates


def _analyze_terrain_colors(frame: np.ndarray, player_coords: List[Tuple[int, int]], player_colors: List[int]) -> Dict[str, Any]:
    """
    Analyze terrain colors to identify walkable area and wall area colors.
    
    Uses the insight that:
    1. The two most frequent colors (excluding player) represent walkable and wall areas
    2. The player is surrounded mostly by walkable area color
    
    Args:
        frame: Current frame (64x64 grid)
        player_coords: List of (row, col) coordinates occupied by player
        player_colors: List of color values at player coordinates
        
    Returns:
        dict: {
            'walkable_area_color': int (0-15) - the walkable terrain color,
            'wall_area_color': int (0-15) - the wall/boundary color,
            'terrain_confidence': float (0-1) - confidence in terrain detection
        }
    """
    try:
        # Step 1: Count all color frequencies, excluding player colors
        unique_player_colors = set(player_colors)
        all_colors = frame.flatten()
        
        # Count colors excluding player colors
        terrain_color_counts = {}
        for color in all_colors:
            color = int(color)
            if color not in unique_player_colors:
                terrain_color_counts[color] = terrain_color_counts.get(color, 0) + 1
        
        if len(terrain_color_counts) < 2:
            # Not enough terrain colors to analyze
            logger.debug("Insufficient terrain colors for analysis")
            return _create_failed_terrain_result()
        
        # Step 2: Find the two most frequent terrain colors
        sorted_colors = sorted(terrain_color_counts.items(), key=lambda x: x[1], reverse=True)
        color1, count1 = sorted_colors[0]
        color2, count2 = sorted_colors[1]
        
        logger.debug(f"Top terrain colors: {color1} ({count1} pixels), {color2} ({count2} pixels)")
        
        # Step 3: Determine which is walkable vs wall by analyzing player surroundings
        walkable_color, wall_color, confidence = _distinguish_walkable_wall(
            frame, player_coords, color1, color2
        )
        
        return {
            'walkable_area_color': walkable_color,
            'wall_area_color': wall_color, 
            'terrain_confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Exception in terrain color analysis: {e}")
        return _create_failed_terrain_result()


def _distinguish_walkable_wall(frame: np.ndarray, player_coords: List[Tuple[int, int]], 
                              color1: int, color2: int) -> Tuple[int, int, float]:
    """
    Distinguish which of two dominant colors is walkable area vs wall area.
    
    Uses boundary-based sampling: examines terrain cells immediately adjacent to player boundary.
    The insight is that the player is surrounded mostly by walkable area color.
    
    Args:
        frame: Current frame
        player_coords: Player coordinate list
        color1, color2: The two most frequent terrain colors
        
    Returns:
        tuple: (walkable_color, wall_color, confidence)
    """
    try:
        if not player_coords:
            return color1, color2, 0.0
        
        # Step 1: Find player boundary cells
        boundary_cells = _find_player_boundary_cells(player_coords, frame.shape)
        if not boundary_cells:
            logger.debug("No boundary cells found - using default assignment")
            return color1, color2, 0.5
        
        # Step 2: Sample terrain colors adjacent to boundary
        adjacent_terrain_colors = _sample_adjacent_terrain(boundary_cells, player_coords, frame)
        if not adjacent_terrain_colors:
            logger.debug("No adjacent terrain found - using default assignment") 
            return color1, color2, 0.5
        
        # Step 3: Count occurrences of color1 vs color2 in adjacent terrain
        color1_count = adjacent_terrain_colors.count(color1)
        color2_count = adjacent_terrain_colors.count(color2)
        total_relevant = color1_count + color2_count
        total_sampled = len(adjacent_terrain_colors)
        
        logger.debug(f"Adjacent terrain sampling: {total_sampled} cells, color1({color1}): {color1_count}, color2({color2}): {color2_count}")
        
        if total_relevant == 0:
            # Neither dominant color found in adjacent terrain
            logger.debug("Neither dominant color found adjacent to player - using default assignment")
            return color1, color2, 0.5
        
        # Step 4: The color that appears more adjacent to the player is walkable
        if color1_count > color2_count:
            walkable_color, wall_color = color1, color2
            confidence = color1_count / total_relevant
        else:
            walkable_color, wall_color = color2, color1
            confidence = color2_count / total_relevant
        
        # Step 5: Calculate sophisticated confidence based on boundary coverage
        # Higher confidence when most adjacent terrain is one type
        boundary_coverage = total_relevant / total_sampled if total_sampled > 0 else 0
        
        # Adjust confidence based on how much of adjacent terrain is the dominant colors
        if boundary_coverage >= 0.8:
            # Most adjacent terrain is one of the dominant colors - boost confidence
            confidence = min(1.0, confidence + 0.1)
        elif boundary_coverage < 0.5:
            # Too much other terrain around - lower confidence
            confidence *= 0.8
        
        logger.debug(f"Boundary terrain analysis: walkable={walkable_color}, wall={wall_color}, "
                    f"confidence={confidence:.3f}, coverage={boundary_coverage:.3f}")
        
        return walkable_color, wall_color, confidence
        
    except Exception as e:
        logger.error(f"Exception in walkable/wall distinction: {e}")
        return color1, color2, 0.0


def _find_player_boundary_cells(player_coords: List[Tuple[int, int]], frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find player cells that are on the boundary (perimeter) of the player shape.
    
    A player cell is a boundary cell if it has at least one non-player neighbor.
    
    Args:
        player_coords: List of (row, col) coordinates occupied by player
        frame_shape: Shape of the frame (height, width) for bounds checking
        
    Returns:
        List of (row, col) coordinates that are on the player boundary
    """
    if not player_coords:
        return []
        
    player_coord_set = set(player_coords)
    boundary_cells = []
    height, width = frame_shape
    
    for row, col in player_coords:
        # Check if this player cell has any non-player neighbors (4-connected)
        is_boundary = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
            neighbor_row, neighbor_col = row + dr, col + dc
            
            # Check if neighbor is within bounds
            if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                # If neighbor is not a player cell, this is a boundary cell
                if (neighbor_row, neighbor_col) not in player_coord_set:
                    is_boundary = True
                    break
            else:
                # Out of bounds neighbors also make this a boundary cell
                is_boundary = True
                break
        
        if is_boundary:
            boundary_cells.append((row, col))
    
    logger.debug(f"Found {len(boundary_cells)} boundary cells out of {len(player_coords)} total player cells")
    return boundary_cells


def _sample_adjacent_terrain(boundary_cells: List[Tuple[int, int]], 
                            player_coords: List[Tuple[int, int]], 
                            frame: np.ndarray) -> List[int]:
    """
    Sample terrain cells that are immediately adjacent to the player boundary.
    
    For each boundary cell, examine its 4-connected neighbors that are not player positions.
    
    Args:
        boundary_cells: List of player cells on the boundary
        player_coords: Complete list of player coordinates (for exclusion)
        frame: Current frame to sample colors from
        
    Returns:
        List of color values from terrain cells adjacent to player boundary
    """
    if not boundary_cells:
        return []
        
    player_coord_set = set(player_coords)
    adjacent_terrain_colors = []
    height, width = frame.shape
    sampled_coords = set()  # Avoid double-counting same terrain cell
    
    for row, col in boundary_cells:
        # Check all 4-connected neighbors of this boundary cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
            neighbor_row, neighbor_col = row + dr, col + dc
            
            # Check bounds and ensure it's not a player cell
            if (0 <= neighbor_row < height and 0 <= neighbor_col < width and
                (neighbor_row, neighbor_col) not in player_coord_set):
                
                # Avoid sampling the same terrain cell multiple times
                coord = (neighbor_row, neighbor_col)
                if coord not in sampled_coords:
                    sampled_coords.add(coord)
                    color = int(frame[neighbor_row, neighbor_col])
                    adjacent_terrain_colors.append(color)
    
    logger.debug(f"Sampled {len(adjacent_terrain_colors)} adjacent terrain cells from {len(boundary_cells)} boundary cells")
    return adjacent_terrain_colors


def _create_failed_terrain_result() -> Dict[str, Any]:
    """Create failed terrain analysis result."""
    return {
        'walkable_area_color': -1,  # Invalid color to indicate failure
        'wall_area_color': -1,
        'terrain_confidence': 0.0
    }


def _create_no_movement_result() -> Dict[str, Any]:
    """Create result for case where no movement was detected."""
    return {
        'player_coords': [],
        'movement_direction': 0,
        'player_size': 0,
        'confidence': 0.0,
        'old_player_coords': [],
        'energy_consumption': 0,
        'total_changes': 0,
        'player_colors': [],
        'old_player_colors': [],
        'walkable_area_color': -1,
        'wall_area_color': -1,
        'terrain_confidence': 0.0,
        'success': False,
        'reason': 'no_movement'
    }


def _create_failed_result(total_changes: int) -> Dict[str, Any]:
    """Create result for case where extraction failed."""
    return {
        'player_coords': [],
        'movement_direction': 0,
        'player_size': 0,
        'confidence': 0.0,
        'old_player_coords': [],
        'energy_consumption': 0,
        'total_changes': total_changes,
        'player_colors': [],
        'old_player_colors': [],
        'walkable_area_color': -1,
        'wall_area_color': -1,
        'terrain_confidence': 0.0,
        'success': False,
        'reason': 'extraction_failed'
    }


def simulate_and_visualize_player_actions(frame_1: np.ndarray, 
                                        player_coords: List[Tuple[int, int]], 
                                        player_colors: List[int],
                                        walkable_area_color: int, 
                                        wall_area_color: int, 
                                        actions: List[int]) -> plt.Figure:
    """
    Simulate player actions and create a matplotlib visualization showing the result.
    
    Args:
        frame_1: Initial frame (64x64 grid)
        player_coords: List of (row, col) coordinates currently occupied by player
        player_colors: List of color values at player coordinates
        walkable_area_color: Color value (0-15) for walkable terrain
        wall_area_color: Color value (0-15) for wall terrain
        actions: List of action numbers to simulate (1=up, 2=down, 3=left, 4=right)
        
    Returns:
        matplotlib Figure showing the simulation result
    """
    try:
        # Step 1: Simulate player movement
        final_player_coords, simulation_valid = _simulate_player_actions(
            frame_1, player_coords, walkable_area_color, wall_area_color, actions
        )
        
        # Step 2: Create visualization
        fig = _visualize_player_simulation(
            frame_1, player_coords, final_player_coords, player_colors, 
            walkable_area_color, wall_area_color, actions, simulation_valid
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Exception in player action simulation: {e}")
        # Return a basic error visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


def _simulate_player_actions(frame: np.ndarray, 
                           initial_coords: List[Tuple[int, int]],
                           walkable_color: int, 
                           wall_color: int, 
                           actions: List[int]) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate player movement through a sequence of actions.
    
    Args:
        frame: Current game frame
        initial_coords: Starting player coordinates
        walkable_color: Color that player can move through
        wall_color: Color that blocks player movement
        actions: List of actions (1=up, 2=down, 3=left, 4=right)
        
    Returns:
        Tuple of (final_coordinates, simulation_successful)
    """
    if not initial_coords or not actions:
        return initial_coords, True
    
    # Action to direction mapping
    action_to_direction = {
        1: (-1, 0),  # Up
        2: (1, 0),   # Down
        3: (0, -1),  # Left
        4: (0, 1),   # Right
    }
    
    current_coords = initial_coords.copy()
    height, width = frame.shape
    
    logger.debug(f"Starting simulation with {len(current_coords)} player cells and {len(actions)} actions")
    
    for action_idx, action in enumerate(actions):
        if action not in action_to_direction:
            logger.debug(f"Skipping invalid action {action} at step {action_idx}")
            continue
            
        dr, dc = action_to_direction[action]
        
        # Calculate new position for all player cells
        new_coords = []
        valid_move = True
        
        for row, col in current_coords:
            new_row, new_col = row + dr, col + dc
            
            # Check boundaries
            if not (0 <= new_row < height and 0 <= new_col < width):
                valid_move = False
                break
                
            # Check for wall collision
            if frame[new_row, new_col] == wall_color:
                valid_move = False
                break
                
            new_coords.append((new_row, new_col))
        
        if valid_move:
            current_coords = new_coords
            logger.debug(f"Action {action} successful: moved player")
        else:
            logger.debug(f"Action {action} blocked: wall collision or boundary")
            
    logger.debug(f"Simulation complete: final position has {len(current_coords)} cells")
    return current_coords, True


def _visualize_player_simulation(frame: np.ndarray,
                               initial_coords: List[Tuple[int, int]], 
                               final_coords: List[Tuple[int, int]],
                               player_colors: List[int],
                               walkable_color: int,
                               wall_color: int,
                               actions: List[int],
                               simulation_valid: bool) -> plt.Figure:
    """
    Create matplotlib visualization of the player simulation.
    
    Args:
        frame: Original game frame
        initial_coords: Starting player position
        final_coords: Final player position after actions
        player_colors: Colors to use for player visualization
        walkable_color: Walkable terrain color
        wall_color: Wall terrain color
        actions: List of actions that were simulated
        simulation_valid: Whether simulation completed successfully
        
    Returns:
        matplotlib Figure with the visualization
    """
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Fill old player position with walkable area color
    for row, col in initial_coords:
        vis_frame[row, col] = walkable_color
    
    # Place player at final position with their colors
    player_color_cycle = player_colors if player_colors else [9]  # Default to red if no colors
    for idx, (row, col) in enumerate(final_coords):
        color = player_color_cycle[idx % len(player_color_cycle)]
        vis_frame[row, col] = color
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # ARC-AGI-3 color palette (0-15)
    arc_colors = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue  
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: sky
        '#870C25',  # 9: brown
        '#FDBCB4',  # 10: pink
        '#8FB339',  # 11: lime
        '#40E0D0',  # 12: teal
        '#001F3F',  # 13: navy
        '#800000',  # 14: maroon
        '#FFFFFF'   # 15: white
    ]
    
    # Create custom colormap
    cmap = mcolors.ListedColormap(arc_colors)
    bounds = list(range(17))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Display the frame
    im = ax.imshow(vis_frame, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Remove all titles, labels, ticks, and legends for clean visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    return fig