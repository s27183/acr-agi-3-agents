"""
Tests for Autonomous Player Coordinate Extractor

Tests various player shapes, movement directions, and edge cases
to ensure robust player detection across different scenarios.
"""

import pytest
import numpy as np
from agents.rl.autonomous_player_extractor import (
    extract_player_coordinates,
    find_connected_components,
    find_optimal_player_pair,
    calculate_size_similarity,
    calculate_spatial_proximity,
    calculate_dominance_ratio,
    calculate_movement_alignment,
    infer_movement_direction,
    Component,
    ComponentPair
)


class TestConnectedComponents:
    """Test connected component detection."""
    
    def test_single_component(self):
        """Test detection of single connected component."""
        diff_grid = np.zeros((5, 5), dtype=np.int32)
        diff_grid[1:3, 1:3] = 1  # 2x2 square
        
        components = find_connected_components(diff_grid)
        
        assert len(components) == 1
        assert components[0].size == 4
        assert set(components[0].coordinates) == {(1,1), (1,2), (2,1), (2,2)}
    
    def test_multiple_components(self):
        """Test detection of multiple separate components."""
        diff_grid = np.zeros((6, 6), dtype=np.int32)
        diff_grid[1, 1] = 1      # Single cell
        diff_grid[3:5, 3:5] = 2  # 2x2 square
        
        components = find_connected_components(diff_grid)
        
        assert len(components) == 2
        # Should be sorted by size (largest first)
        assert components[0].size == 4  # 2x2 square
        assert components[1].size == 1  # Single cell
    
    def test_l_shaped_component(self):
        """Test detection of L-shaped player."""
        diff_grid = np.zeros((5, 5), dtype=np.int32)
        # Create L-shape
        diff_grid[1, 1:4] = 1  # Horizontal bar
        diff_grid[2:4, 1] = 1  # Vertical bar
        
        components = find_connected_components(diff_grid)
        
        assert len(components) == 1
        assert components[0].size == 5  # 3 + 2 cells (overlap at (1,1))
    
    def test_empty_grid(self):
        """Test with no changes."""
        diff_grid = np.zeros((5, 5), dtype=np.int32)
        
        components = find_connected_components(diff_grid)
        
        assert len(components) == 0


class TestComponentScoring:
    """Test component pair scoring functions."""
    
    def test_size_similarity_identical(self):
        """Test size similarity for identical components."""
        comp1 = Component(coordinates=[(0,0), (0,1)])
        comp2 = Component(coordinates=[(2,2), (2,3)])
        
        similarity = calculate_size_similarity(comp1, comp2)
        
        assert similarity == 1.0
    
    def test_size_similarity_different(self):
        """Test size similarity for different sized components."""
        comp1 = Component(coordinates=[(0,0)])  # Size 1
        comp2 = Component(coordinates=[(2,2), (2,3), (3,2)])  # Size 3
        
        similarity = calculate_size_similarity(comp1, comp2)
        
        assert similarity == 1.0 - (2/3)  # 1 - |1-3|/max(1,3)
    
    def test_spatial_proximity_adjacent(self):
        """Test spatial proximity for adjacent components."""
        comp1 = Component(coordinates=[(1,1), (1,2)])
        comp2 = Component(coordinates=[(1,3), (1,4)])  # Adjacent horizontally
        
        proximity = calculate_spatial_proximity(comp1, comp2)
        
        assert proximity > 0.5  # Should be high for adjacent components
    
    def test_spatial_proximity_overlapping(self):
        """Test spatial proximity for overlapping bounding boxes."""
        comp1 = Component(coordinates=[(1,1), (1,2)])
        comp2 = Component(coordinates=[(0,2), (2,2)])  # Overlapping bbox
        
        proximity = calculate_spatial_proximity(comp1, comp2)
        
        assert proximity == 1.0  # Perfect for overlapping bounding boxes
    
    def test_dominance_ratio_majority(self):
        """Test dominance ratio when pair dominates changes."""
        comp1 = Component(coordinates=[(0,0), (0,1)])  # Size 2
        comp2 = Component(coordinates=[(1,0), (1,1)])  # Size 2
        total_changes = 5  # 4/5 = 0.8 dominance
        
        dominance = calculate_dominance_ratio(comp1, comp2, total_changes)
        
        assert dominance == 0.8
    
    def test_movement_alignment_horizontal(self):
        """Test movement alignment for horizontal movement."""
        comp1 = Component(coordinates=[(2,1), (2,2)])  # Centroid (2, 1.5)
        comp2 = Component(coordinates=[(2,4), (2,5)])  # Centroid (2, 4.5)
        
        alignment = calculate_movement_alignment(comp1, comp2)
        
        assert alignment == 1.0  # Perfect horizontal alignment


class TestPlayerExtraction:
    """Test complete player coordinate extraction."""
    
    def test_simple_horizontal_movement(self):
        """Test extraction of simple horizontal movement."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = np.zeros((10, 10), dtype=np.int32)
        
        # Player moves from (5,2)-(5,3) to (5,4)-(5,5)
        frame_0[5, 2:4] = 5  # Old position
        frame_1[5, 4:6] = 5  # New position
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        assert result['movement_direction'] == 4  # Right
        assert result['player_size'] == 2
        assert set(result['player_coords']) == {(5,4), (5,5)}
        assert set(result['old_player_coords']) == {(5,2), (5,3)}
        assert result['energy_consumption'] == 0  # 4 total changes - 2*2 player = 0
    
    def test_vertical_movement_with_energy(self):
        """Test extraction with energy consumption."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = np.zeros((10, 10), dtype=np.int32)
        
        # Player moves vertically: 2x2 square moves up
        frame_0[5:7, 3:5] = 3  # Old 2x2 position
        frame_1[3:5, 3:5] = 3  # New 2x2 position
        
        # Add energy consumption (e.g., energy bar change)
        frame_1[0, 0] = 7  # Energy consumption change
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        assert result['movement_direction'] == 1  # Up
        assert result['player_size'] == 4  # 2x2 = 4 cells
        assert result['energy_consumption'] == 1  # 9 total - 2*4 = 1
        assert len(result['player_coords']) == 4
        assert len(result['old_player_coords']) == 4
    
    def test_l_shaped_player(self):
        """Test extraction of L-shaped player."""
        frame_0 = np.zeros((8, 8), dtype=np.int32)
        frame_1 = np.zeros((8, 8), dtype=np.int32)
        
        # L-shaped player moves right
        # Old position: L-shape at (2,1)
        frame_0[2, 1:4] = 4  # Horizontal bar
        frame_0[3:5, 1] = 4  # Vertical bar
        
        # New position: Same L-shape at (2,2)
        frame_1[2, 2:5] = 4  # Horizontal bar moved right
        frame_1[3:5, 2] = 4  # Vertical bar moved right
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        assert result['movement_direction'] == 4  # Right
        assert result['player_size'] == 5  # L-shape has 5 cells
        assert result['energy_consumption'] == 0
    
    def test_no_movement(self):
        """Test case where no movement occurred."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = frame_0.copy()  # Identical frames
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is False
        assert result['reason'] == 'no_movement'
        assert result['total_changes'] == 0
    
    def test_wall_collision_simulation(self):
        """Test case simulating wall collision (only energy changes)."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = frame_0.copy()
        
        # Only energy bar changes (no player movement)
        frame_1[0, 5:8] = 2  # Energy bar change
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        # Should fail to find player pair since only one component exists
        assert result['success'] is False
        assert result['total_changes'] == 3
    
    def test_complex_background_changes(self):
        """Test with multiple environmental changes."""
        frame_0 = np.zeros((12, 12), dtype=np.int32)
        frame_1 = np.zeros((12, 12), dtype=np.int32)
        
        # Player movement: 3x1 rectangle moves down
        frame_0[2, 5:8] = 6  # Old position
        frame_1[4, 5:8] = 6  # New position
        
        # Multiple environmental changes
        frame_1[0, 0] = 1    # Energy change 1
        frame_1[1, 1:3] = 2  # Energy change 2 (2 cells)
        frame_1[10, 10] = 3  # Energy change 3
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        assert result['movement_direction'] == 2  # Down
        assert result['player_size'] == 3
        assert result['energy_consumption'] == 4  # 10 total - 2*3 = 4
        assert len(result['player_coords']) == 3
    
    def test_ambiguous_case_largest_pair(self):
        """Test case with multiple similar pairs - should pick largest."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = np.zeros((10, 10), dtype=np.int32)
        
        # Create two potential player movements
        # Larger movement (4 cells)
        frame_0[2:4, 2:4] = 5  # 2x2 old
        frame_1[2:4, 5:7] = 5  # 2x2 new
        
        # Smaller movement (2 cells) 
        frame_0[7, 1:3] = 3    # 1x2 old
        frame_1[7, 8:10] = 3   # 1x2 new
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        # Should pick the larger pair (2x2 = 4 cells)
        assert result['player_size'] == 4
        assert result['movement_direction'] == 4  # Right


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_large_component(self):
        """Test with only one large component (no pair)."""
        frame_0 = np.zeros((8, 8), dtype=np.int32)
        frame_1 = np.zeros((8, 8), dtype=np.int32)
        
        # Large single change (no clear old/new pair)
        frame_1[2:6, 2:6] = 7  # 4x4 change
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is False
        assert result['total_changes'] == 16
    
    def test_many_small_components(self):
        """Test with many small scattered changes."""
        frame_0 = np.zeros((10, 10), dtype=np.int32)
        frame_1 = np.zeros((10, 10), dtype=np.int32)
        
        # Many scattered single-cell changes
        for i in range(8):
            frame_1[i, i] = i + 1
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        # Should fail due to no similar-sized pairs
        assert result['success'] is False
    
    def test_minimal_valid_case(self):
        """Test minimal valid case (2 single-cell components)."""
        frame_0 = np.zeros((5, 5), dtype=np.int32)
        frame_1 = np.zeros((5, 5), dtype=np.int32)
        
        # Single cell player moves
        frame_0[2, 1] = 5
        frame_1[2, 3] = 5
        
        result = extract_player_coordinates(frame_0, frame_1)
        
        assert result['success'] is True
        assert result['player_size'] == 1
        assert result['movement_direction'] == 4  # Right
        assert result['energy_consumption'] == 0


@pytest.fixture
def sample_components():
    """Fixture providing sample components for testing."""
    comp1 = Component(coordinates=[(1,1), (1,2)])
    comp2 = Component(coordinates=[(3,1), (3,2)])
    comp3 = Component(coordinates=[(5,5)])
    return [comp1, comp2, comp3]


def test_component_dataclass():
    """Test Component dataclass functionality."""
    coords = [(0,0), (0,1), (1,0)]
    comp = Component(coordinates=coords)
    
    assert comp.size == 3
    assert comp.centroid == (1/3, 1/3)  # Average of coordinates
    assert comp.bounding_box['min_row'] == 0
    assert comp.bounding_box['max_row'] == 1
    assert comp.bounding_box['min_col'] == 0
    assert comp.bounding_box['max_col'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])