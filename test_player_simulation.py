#!/usr/bin/env python3

"""
Test the player action simulation and visualization functionality.
"""

import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from agents.rl.autonomous_player_extractor import (
    extract_player_coordinates,
    simulate_and_visualize_player_actions
)

# Enable debug logging to see detailed output
logging.basicConfig(level=logging.INFO)

def test_player_simulation():
    """Test the player action simulation with real game data."""
    
    print("="*60)
    print("TESTING PLAYER ACTION SIMULATION")
    print("="*60)
    
    # Load real game data from our previous test
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
    print(f"Movement direction detected: {result['movement_direction']} (left)")
    
    # Test various action sequences
    test_cases = [
        {
            'name': 'Move Right (opposite to detected left movement)',
            'actions': [4],  # Right
        },
        {
            'name': 'Move Right then Down',
            'actions': [4, 2],  # Right, Down
        },
        {
            'name': 'Move Up then Left then Down',
            'actions': [1, 3, 2],  # Up, Left, Down
        },
        {
            'name': 'Complex sequence - Up, Right, Right, Down, Left',
            'actions': [1, 4, 4, 2, 3],  # Up, Right, Right, Down, Left
        },
        {
            'name': 'Try to hit walls (should be blocked)',
            'actions': [3, 3, 3, 3, 3],  # Multiple left moves (should hit walls)
        }
    ]
    
    # Run each test case
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        print(f"Actions: {test_case['actions']}")
        
        try:
            # Run simulation
            fig = simulate_and_visualize_player_actions(
                frame_1=frame_1,
                player_coords=result['player_coords'],
                player_colors=result['player_colors'],
                walkable_area_color=result['walkable_area_color'],
                wall_area_color=result['wall_area_color'],
                actions=test_case['actions']
            )
            
            # Save the visualization
            filename = f"player_simulation_test_{i+1}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Simulation successful - saved to {filename}")
            plt.close(fig)  # Clean up
            
        except Exception as e:
            print(f"✗ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("SIMULATION TESTS COMPLETE")
    print("Check the generated PNG files to see the visualizations!")
    print("="*60)

if __name__ == "__main__":
    test_player_simulation()