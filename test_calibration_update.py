#!/usr/bin/env python3

"""
Test that the calibrate_at_reset() method now uses extract_player_coordinates() function.
"""

import json
import numpy as np
import logging
from agents.rl.calibration import Calibration
from agents.structs import FrameData

# Enable debug logging
logging.basicConfig(level=logging.INFO)

def create_frame_data(grid_data, score=0):
    """Helper to create FrameData objects"""
    return FrameData(
        frame=[grid_data],
        score=score,
        state='NOT_FINISHED',
        game_id='test',
        action_input={'id': 0, 'reasoning': 'test'},
        timestamp='2024-01-01T00:00:00'
    )

def test_calibration_uses_autonomous_extraction():
    """Test that calibrate_at_reset uses autonomous extraction"""
    
    # Create mock SwarmOperations
    class MockSwarmOps:
        def __init__(self):
            # Load real game data for testing
            jsonl_file = "recordings/ls20-f340c8e5138e.rlagent.rl.5f81e5fc-d337-4c44-a802-ebcdf7de04e4.recording.jsonl"
            
            with open(jsonl_file, 'r') as f:
                lines = list(f)
                
            # Get frames that have movement (frames 3 and 4 from our previous test)
            reset_data = json.loads(lines[3].strip())
            move_data = json.loads(lines[4].strip())
            
            self.last_frame = create_frame_data(reset_data["data"]["frame"][0], reset_data["data"]["score"])
            self.test_frame = create_frame_data(move_data["data"]["frame"][0], move_data["data"]["score"])
            self.action_count = 0
            
        def execute_action(self, action_id):
            self.action_count += 1
            if self.action_count == 1:
                # Return the movement frame on first action
                return self.test_frame
            return None
    
    print("Testing calibrate_at_reset() with autonomous extraction...")
    
    # Create calibration and mock swarm ops
    calibration = Calibration()
    swarm_ops = MockSwarmOps()
    
    # Run calibration
    success, frame = calibration.calibrate_at_reset(swarm_ops)
    
    print(f"\nCalibration result: success={success}")
    if success:
        print(f"Player size: {calibration.player_size}")
        print(f"Energy consumption: {calibration.energy_consumption}")
        print(f"Player location: {calibration.player_location}")
        print(f"Calibrated: {calibration.calibrated}")
    
    return success

if __name__ == "__main__":
    test_calibration_uses_autonomous_extraction()