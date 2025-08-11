#!/usr/bin/env python3

"""
Test the autonomous player extractor with real game data from recordings.
"""

import json
import numpy as np
import logging
from agents.rl.autonomous_player_extractor import extract_player_coordinates

# Enable debug logging to see detailed output
logging.basicConfig(level=logging.DEBUG)

def load_first_two_frames(jsonl_file_path):
    """
    Load the first two responses from a JSONL recording file that have actual movement.
    
    Args:
        jsonl_file_path: Path to the JSONL recording file
        
    Returns:
        tuple: (frame_0, frame_1) as numpy arrays, or None if no movement found
    """
    frames = []
    frame_info = []
    
    with open(jsonl_file_path, 'r') as f:
        for i, line in enumerate(f):
            response = json.loads(line.strip())
            frame_data = response["data"]["frame"][0]  # Get the 64x64 grid
            frame = np.array(frame_data, dtype=np.int32)
            
            info = {
                'index': i,
                'game_id': response['data']['game_id'],
                'timestamp': response['timestamp'],
                'action': response['data']['action_input']['id'],
                'reasoning': response['data']['action_input']['reasoning'],
                'state': response['data']['state'],
                'score': response['data']['score']
            }
            
            frames.append(frame)
            frame_info.append(info)
            
            # Check for movement if we have at least 2 frames
            if len(frames) >= 2:
                diff_grid = frames[-1].astype(np.int32) - frames[-2].astype(np.int32)
                total_changes = np.sum(diff_grid != 0)
                
                if total_changes > 0:
                    print(f"Found movement between frames {i-1} and {i}:")
                    print(f"  Frame {i-1}: {frame_info[-2]['action']} - {frame_info[-2]['reasoning']}")
                    print(f"  Frame {i}: {frame_info[-1]['action']} - {frame_info[-1]['reasoning']}")
                    print(f"  Total changes: {total_changes} cells")
                    return frames[-2], frames[-1]
            
            # Safety limit - don't read too many frames
            if i > 20:
                break
    
    print("No movement detected in first 20 frames")
    return None, None

def analyze_frame_differences(frame_0, frame_1):
    """Analyze the differences between two frames."""
    diff_grid = frame_1.astype(np.int32) - frame_0.astype(np.int32)
    changed_cells = (diff_grid != 0)
    total_changes = np.sum(changed_cells)
    
    print(f"\n=== Frame Difference Analysis ===")
    print(f"Total changed cells: {total_changes}")
    
    if total_changes > 0:
        # Find regions with changes
        changed_coords = np.where(changed_cells)
        min_row, max_row = np.min(changed_coords[0]), np.max(changed_coords[0])
        min_col, max_col = np.min(changed_coords[1]), np.max(changed_coords[1])
        
        print(f"Changed region: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        
        # Show the change region
        region_height = max_row - min_row + 3
        region_width = max_col - min_col + 3
        
        start_row = max(0, min_row - 1)
        end_row = min(64, max_row + 2)
        start_col = max(0, min_col - 1) 
        end_col = min(64, max_col + 2)
        
        print(f"\nFrame 0 change region (rows {start_row}-{end_row-1}, cols {start_col}-{end_col-1}):")
        print(frame_0[start_row:end_row, start_col:end_col])
        
        print(f"\nFrame 1 change region (rows {start_row}-{end_row-1}, cols {start_col}-{end_col-1}):")
        print(frame_1[start_row:end_row, start_col:end_col])
        
        print(f"\nDifference grid (frame_1 - frame_0):")
        print(diff_grid[start_row:end_row, start_col:end_col])
        
        # Show change values
        unique_changes = np.unique(diff_grid[changed_cells])
        print(f"Unique change values: {unique_changes}")
        
        for value in unique_changes:
            count = np.sum(diff_grid == value)
            print(f"  Value {value}: {count} cells")

def test_real_game_extraction():
    """Test the autonomous player extractor with real game data."""
    
    # Path to the recording file
    jsonl_file = "recordings/ls20-f340c8e5138e.rlagent.rl.5f81e5fc-d337-4c44-a802-ebcdf7de04e4.recording.jsonl"
    
    print("="*60)
    print("TESTING AUTONOMOUS PLAYER EXTRACTOR WITH REAL GAME DATA")
    print("="*60)
    
    # Load the first two frames with movement
    try:
        frame_0, frame_1 = load_first_two_frames(jsonl_file)
        if frame_0 is None or frame_1 is None:
            print("No movement found in recording - cannot test player extraction")
            return
        print(f"\nSuccessfully loaded frames with shapes: {frame_0.shape}, {frame_1.shape}")
    except Exception as e:
        print(f"Error loading frames: {e}")
        return
    
    # Analyze the differences between frames
    analyze_frame_differences(frame_0, frame_1)
    
    # Test the autonomous player extractor
    print(f"\n" + "="*50)
    print("RUNNING AUTONOMOUS PLAYER EXTRACTION")
    print("="*50)
    
    try:
        result = extract_player_coordinates(frame_0, frame_1)
        
        print(f"\nEXTRACTION RESULTS:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"Player coordinates: {len(result['player_coords'])} coordinates")
            print(f"Old player coordinates: {len(result['old_player_coords'])} coordinates")
            print(f"Movement direction: {result['movement_direction']} ", end="")
            direction_names = {0: "unknown", 1: "up", 2: "down", 3: "left", 4: "right"}
            print(f"({direction_names.get(result['movement_direction'], 'unknown')})")
            print(f"Player size: {result['player_size']} cells")
            print(f"Energy consumption: {result['energy_consumption']} cells")
            print(f"Total changes: {result['total_changes']} cells")
            print(f"Confidence: {result['confidence']:.3f}")
            
            # Show player colors
            print(f"\nPlayer colors (current): {sorted(set(result['player_colors']))}")
            print(f"Old player colors: {sorted(set(result['old_player_colors']))}")
            
            # Count color frequency
            from collections import Counter
            current_colors = Counter(result['player_colors'])
            old_colors = Counter(result['old_player_colors'])
            print(f"Current color distribution: {dict(current_colors)}")
            print(f"Old color distribution: {dict(old_colors)}")
            
            # Show terrain analysis
            print(f"\nTerrain Analysis:")
            print(f"Walkable area color: {result['walkable_area_color']}")
            print(f"Wall area color: {result['wall_area_color']}")
            print(f"Terrain confidence: {result['terrain_confidence']:.3f}")
            
            # Color meanings for reference
            color_names = {
                0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "gray",
                6: "magenta", 7: "orange", 8: "sky", 9: "brown", 10: "pink", 11: "lime",
                12: "teal", 13: "navy", 14: "maroon", 15: "white"
            }
            walkable_name = color_names.get(result['walkable_area_color'], 'unknown')
            wall_name = color_names.get(result['wall_area_color'], 'unknown') 
            print(f"Walkable area: {walkable_name} ({result['walkable_area_color']})")
            print(f"Wall area: {wall_name} ({result['wall_area_color']})")
        else:
            print(f"Reason: {result.get('reason', 'unknown')}")
            print(f"Total changes: {result.get('total_changes', 0)}")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_game_extraction()