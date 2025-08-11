#!/usr/bin/env python3

"""
Test custom action sequence: [up, up, right, right, right, up, up, right, right, left]
"""

import json
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from agents.rl.autonomous_player_extractor import (
    extract_player_coordinates,
    simulate_and_visualize_player_actions
)

# Enable logging
logging.basicConfig(level=logging.INFO)

def test_custom_action_sequence():
    """Test the specific action sequence requested by the user."""
    
    print("="*60)
    print("TESTING CUSTOM ACTION SEQUENCE")
    print("Actions: [up, up, right, right, right, up, up, right, right, left]")
    print("="*60)
    
    # Load real game data
    jsonl_file = "recordings/ls20-f340c8e5138e.rlagent.rl.5f81e5fc-d337-4c44-a802-ebcdf7de04e4.recording.jsonl"
    
    # Load frames with movement
    frames = []
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Only need first few lines
                break
            response = json.loads(line.strip())
            frame_data = response["data"]["frame"][0]  # Get the 64x64 grid
            frames.append(np.array(frame_data, dtype=np.int32))
    
    # Use frames 3 and 4 which we know have movement
    frame_0, frame_1 = frames[3], frames[4]
    
    # Extract player information
    print("Extracting player information...")
    result = extract_player_coordinates(frame_0, frame_1)
    
    if not result['success']:
        print(f"Failed to extract player info: {result.get('reason', 'unknown')}")
        return
    
    # Show extracted information
    print(f"Player size: {result['player_size']} cells")
    print(f"Player colors: {sorted(set(result['player_colors']))}")
    print(f"Walkable area color: {result['walkable_area_color']} (green)")
    print(f"Wall area color: {result['wall_area_color']} (yellow)")
    
    # Define the custom action sequence
    # 1=up, 2=down, 3=left, 4=right
    custom_actions = [1, 1, 4, 4, 4, 1, 1, 4, 4, 3]  # up, up, right, right, right, up, up, right, right, left
    action_names = ['up', 'up', 'right', 'right', 'right', 'up', 'up', 'right', 'right', 'left']
    
    print(f"Running simulation with actions: {action_names}")
    
    try:
        # Run simulation
        fig = simulate_and_visualize_player_actions(
            frame_1=frame_1,
            player_coords=result['player_coords'],
            player_colors=result['player_colors'],
            walkable_area_color=result['walkable_area_color'],
            wall_area_color=result['wall_area_color'],
            actions=custom_actions
        )
        
        # Create output directory if it doesn't exist
        output_dir = "agents/rl"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the visualization
        filename = os.path.join(output_dir, "custom_action_sequence_simulation.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Simulation successful - saved to {filename}")
        plt.close(fig)  # Clean up
        
        # Also show the absolute path
        abs_path = os.path.abspath(filename)
        print(f"Full path: {abs_path}")
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("CUSTOM ACTION SEQUENCE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_custom_action_sequence()