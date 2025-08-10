import time
from typing import Any, List

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class DeterministicTester(Agent):
    """
    A deterministic agent for testing action effectiveness in real game-playing scenarios.
    
    This agent follows a predetermined sequence of actions to test:
    1. Sequential action application (action i applied to result of action i-1)
    2. Real game state progression
    3. Action effectiveness in context of actual gameplay
    
    Usage:
    - Set action_sequence to test specific action patterns
    - Observes frame changes through actual game progression
    - Provides detailed logging of state transitions
    """

    MAX_ACTIONS = 50

    def __init__(self, action_sequence: List[int] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        # Default test sequence: test each action type in sequence
        self.action_sequence = action_sequence or [1, 2, 3, 4, 5, 1, 3, 4, 2, 5]
        self.current_step = 0
        self.frame_history = []
        self.change_log = []
        
        print(f"🎯 Deterministic Tester initialized")
        print(f"   📋 Test sequence: {self.action_sequence}")
        print(f"   🎮 Game: {self.game_id}")

    @property
    def name(self) -> str:
        return f"{super().name}.deterministic"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Agent is done when sequence is complete or game ends."""
        sequence_complete = self.current_step >= len(self.action_sequence)
        game_won = latest_frame.state is GameState.WIN
        game_over = latest_frame.state is GameState.GAME_OVER
        
        if sequence_complete:
            print(f"✅ Test sequence completed after {self.current_step} steps")
            self._print_analysis_summary()
            
        return sequence_complete or game_won or game_over

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose the next predetermined action in sequence."""
        
        # Handle initial reset only once, or after game over
        if latest_frame.state is GameState.NOT_PLAYED and self.current_step == 0:
            print(f"🔄 RESET: Initializing game state")
            return GameAction.RESET
        elif latest_frame.state is GameState.GAME_OVER:
            print(f"🔄 RESET: Game over, resetting")
            return GameAction.RESET
            
        # Get next action from sequence
        if self.current_step < len(self.action_sequence):
            action_id = self.action_sequence[self.current_step]
            action = GameAction.from_id(action_id)
            
            # Store previous frame for comparison
            if len(frames) >= 2:
                prev_frame = frames[-2]
                curr_frame = frames[-1]
                self._analyze_frame_change(prev_frame, curr_frame, self.current_step - 1)
            
            print(f"🎯 Step {self.current_step + 1}: Executing ACTION{action_id}")
            
            # Set reasoning for action
            action.reasoning = {
                "step": self.current_step + 1,
                "action_id": action_id,
                "test_type": "deterministic_sequence",
                "expected_behavior": "sequential_state_progression"
            }
            
            self.current_step += 1
            return action
        else:
            # Fallback - should not reach here due to is_done check
            print(f"⚠️ Sequence completed, returning RESET")
            return GameAction.RESET

    def _analyze_frame_change(self, prev_frame: FrameData, curr_frame: FrameData, step: int) -> None:
        """Analyze the change between two consecutive frames."""
        if not prev_frame.frame or not curr_frame.frame:
            return
            
        import numpy as np
        
        prev_grid = np.array(prev_frame.frame[0])
        curr_grid = np.array(curr_frame.frame[0])
        
        pixel_changes = np.sum(prev_grid != curr_grid)
        score_change = curr_frame.score - prev_frame.score
        
        action_id = self.action_sequence[step] if step < len(self.action_sequence) else "RESET"
        
        change_info = {
            "step": step + 1,
            "action_id": action_id,
            "pixel_changes": int(pixel_changes),
            "score_change": score_change,
            "prev_score": prev_frame.score,
            "curr_score": curr_frame.score,
            "game_state": curr_frame.state.value
        }
        
        self.change_log.append(change_info)
        
        # Real-time logging
        status = "🟢 EFFECTIVE" if pixel_changes > 0 else "⚪ NO-OP"
        print(f"   {status} ACTION{action_id}: {pixel_changes} pixel changes, score: {prev_frame.score}→{curr_frame.score}")

    def _print_analysis_summary(self) -> None:
        """Print comprehensive analysis of the test sequence."""
        print(f"\n📊 DETERMINISTIC ACTION TEST ANALYSIS")
        print("=" * 60)
        
        if not self.change_log:
            print("❌ No frame changes recorded")
            return
            
        # Effectiveness analysis
        effective_actions = []
        no_op_actions = []
        
        for change in self.change_log:
            action_id = change["action_id"]
            if change["pixel_changes"] > 0:
                effective_actions.append(action_id)
            else:
                no_op_actions.append(action_id)
                
        print(f"🎯 Action Effectiveness Summary:")
        print(f"   ✅ Total actions tested: {len(self.change_log)}")
        print(f"   🟢 Effective actions: {len(effective_actions)} ({len(effective_actions)/len(self.change_log)*100:.1f}%)")
        print(f"   ⚪ No-op actions: {len(no_op_actions)} ({len(no_op_actions)/len(self.change_log)*100:.1f}%)")
        
        # Detailed step-by-step log
        print(f"\n📋 Step-by-Step Analysis:")
        for i, change in enumerate(self.change_log, 1):
            status = "🟢" if change["pixel_changes"] > 0 else "⚪"
            print(f"   {status} Step {change['step']}: ACTION{change['action_id']} → "
                  f"{change['pixel_changes']} pixels, score {change['prev_score']}→{change['curr_score']}")
                  
        # Action frequency analysis
        from collections import Counter
        action_counts = Counter([change["action_id"] for change in self.change_log])
        effective_counts = Counter([change["action_id"] for change in self.change_log if change["pixel_changes"] > 0])
        
        print(f"\n🔢 Action Frequency Analysis:")
        for action_id in sorted(action_counts.keys()):
            total = action_counts[action_id]
            effective = effective_counts.get(action_id, 0)
            effectiveness_rate = (effective / total * 100) if total > 0 else 0
            print(f"   ACTION{action_id}: {effective}/{total} effective ({effectiveness_rate:.1f}%)")


# Specialized tester classes for specific scenarios
class SingleActionTester(DeterministicTester):
    """Test a single action repeatedly to measure consistency."""
    
    def __init__(self, action_id: int, repetitions: int = 10, *args: Any, **kwargs: Any) -> None:
        action_sequence = [action_id] * repetitions
        super().__init__(action_sequence, *args, **kwargs)
        print(f"🔂 Single Action Tester: ACTION{action_id} × {repetitions} repetitions")


class NoOpValidationTester(DeterministicTester):
    """Test suspected no-op actions in real game context."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Test sequence: intersperse suspected no-ops with known effective actions
        action_sequence = [1, 2, 1, 5, 1, 2, 5, 3, 2, 5, 3, 4, 2, 5, 4]
        super().__init__(action_sequence, *args, **kwargs)
        print(f"🔍 No-Op Validation: Testing ACTION2 & ACTION5 in game context")