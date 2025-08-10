import time
from typing import Any

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class ComprehensiveNoOpTest(Agent):
    """
    A comprehensive agent that systematically tests all actions (0-6) across all games
    to validate our isolated testing findings against sequential gameplay patterns.
    
    This agent performs sequential testing to compare with our comprehensive isolated
    analysis, providing definitive answers about action effectiveness in real gameplay.
    
    Test sequence covers:
    - All 7 actions (ACTION0 through ACTION6) 
    - Each action tested multiple times in different contexts
    - Systematic pattern to observe state dependencies
    - Results logged for cross-game analysis
    """

    MAX_ACTIONS = 50  # Comprehensive test sequence

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        # Comprehensive test sequence: each action tested multiple times
        # Pattern: ACTION0→1→2→3→4→5→6→1→0→2→4→3→5→1→6→0→3→2→5→4→6→1→2→3→4→5→0
        self.test_sequence = [
            0, 1, 2, 3, 4, 5, 6,  # First round: all actions in order
            1, 0, 2, 4, 3, 5,     # Second round: effective actions mixed with suspected no-ops  
            1, 6, 0, 3, 2, 5,     # Third round: different pattern
            4, 6, 1, 2, 3, 4,     # Fourth round: focus on context changes
            5, 0, 3, 2, 5, 4,     # Fifth round: validate patterns
            6, 1, 2, 3, 4, 5, 0   # Final round: complete coverage
        ]
        
        self.current_step = 0
        self.change_log = []
        self.initial_reset_done = False
        self.game_start_time = time.time()
        
        print(f"🔍 Comprehensive No-Op Test Agent initialized")
        print(f"🎮 Game: {self.game_id}")
        print(f"📋 Test sequence length: {len(self.test_sequence)} actions")
        print(f"🎯 Testing all actions: ACTION0 through ACTION6")
        print(f"💡 Purpose: Sequential vs Isolated testing comparison")

    @property
    def name(self) -> str:
        return f"{super().name}.comprehensivenooptest"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Agent is done when test sequence is complete or game ends."""
        sequence_complete = self.current_step >= len(self.test_sequence)
        game_won = latest_frame.state is GameState.WIN
        game_over = latest_frame.state is GameState.GAME_OVER
        
        if sequence_complete:
            print(f"✅ Comprehensive test sequence completed after {self.current_step} steps")
            self._print_analysis_summary()
            
        return sequence_complete or game_won or game_over

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose actions systematically to test all action effectiveness."""
        
        # Handle initial reset
        if latest_frame.state is GameState.NOT_PLAYED and not self.initial_reset_done:
            print(f"🔄 RESET: Initializing game state for {self.game_id}")
            self.initial_reset_done = True
            return GameAction.RESET
        elif latest_frame.state is GameState.GAME_OVER:
            print(f"🔄 RESET: Game over, resetting")
            self.initial_reset_done = True
            return GameAction.RESET
            
        # Analyze frame changes from previous action
        if len(frames) >= 2 and self.current_step > 0:
            self._analyze_frame_change(frames[-2], frames[-1], self.current_step - 1)
            
        # Get next action from test sequence
        if self.current_step < len(self.test_sequence):
            action_id = self.test_sequence[self.current_step]
            
            # Map action_id to GameAction - handle ACTION0 as RESET
            if action_id == 0:
                action = GameAction.RESET
                print(f"🎯 Step {self.current_step + 1}: ACTION0 (RESET)")
            else:
                action = GameAction.from_id(action_id)
                
                # Handle ACTION6 with coordinates (center of grid)
                if action_id == 6:
                    action.set_data({"x": 32, "y": 32})
                    print(f"🎯 Step {self.current_step + 1}: ACTION6 (x=32, y=32)")
                else:
                    print(f"🎯 Step {self.current_step + 1}: ACTION{action_id}")
            
            # Set reasoning for action
            action.reasoning = {
                "step": self.current_step + 1,
                "action_id": action_id,
                "test_type": "comprehensive_sequential",
                "game_id": self.game_id,
                "purpose": "validate_all_actions_across_games"
            }
            
            self.current_step += 1
            return action
        else:
            # Fallback - should not reach here due to is_done check
            print(f"⚠️ Test sequence completed, returning RESET")
            return GameAction.RESET

    def _analyze_frame_change(self, prev_frame: FrameData, curr_frame: FrameData, step: int) -> None:
        """Analyze pixel changes between consecutive frames."""
        if not prev_frame.frame or not curr_frame.frame:
            return
            
        import numpy as np
        
        prev_grid = np.array(prev_frame.frame[0])
        curr_grid = np.array(curr_frame.frame[0])
        
        pixel_changes = np.sum(prev_grid != curr_grid)
        score_change = curr_frame.score - prev_frame.score
        
        action_id = self.test_sequence[step] if step < len(self.test_sequence) else "UNKNOWN"
        
        change_info = {
            "step": step + 1,
            "action_id": action_id,
            "pixel_changes": int(pixel_changes),
            "score_change": score_change,
            "prev_score": prev_frame.score,
            "curr_score": curr_frame.score,
            "game_state": curr_frame.state.value,
            "game_id": self.game_id,
            "timestamp": time.time() - self.game_start_time
        }
        
        self.change_log.append(change_info)
        
        # Real-time logging with effectiveness status
        status = "🟢 EFFECTIVE" if pixel_changes > 0 else "⚪ NO-OP"
        print(f"   {status}: ACTION{action_id} → {pixel_changes} pixels, score: {prev_frame.score}→{curr_frame.score}")

    def _print_analysis_summary(self) -> None:
        """Print comprehensive analysis of all actions tested."""
        print(f"\n📊 COMPREHENSIVE SEQUENTIAL TEST ANALYSIS")
        print("=" * 70)
        print(f"🎮 Game: {self.game_id}")
        print(f"⏱️  Duration: {time.time() - self.game_start_time:.1f}s")
        
        if not self.change_log:
            print("❌ No frame changes recorded")
            return
            
        # Overall statistics
        total_actions = len(self.change_log)
        effective_actions = sum(1 for change in self.change_log if change["pixel_changes"] > 0)
        no_op_actions = total_actions - effective_actions
        
        print(f"📈 Overall Statistics:")
        print(f"   🎯 Total actions tested: {total_actions}")
        print(f"   🟢 Effective actions: {effective_actions} ({effective_actions/total_actions*100:.1f}%)")
        print(f"   ⚪ No-op actions: {no_op_actions} ({no_op_actions/total_actions*100:.1f}%)")
        
        # Per-action analysis
        from collections import Counter, defaultdict
        
        action_counts = Counter([change["action_id"] for change in self.change_log])
        effective_counts = Counter([change["action_id"] for change in self.change_log if change["pixel_changes"] > 0])
        
        # Collect pixel change statistics per action
        action_pixel_stats = defaultdict(list)
        for change in self.change_log:
            action_pixel_stats[change["action_id"]].append(change["pixel_changes"])
        
        print(f"\n🔢 Per-Action Analysis:")
        for action_id in sorted(action_counts.keys()):
            total = action_counts[action_id]
            effective = effective_counts.get(action_id, 0)
            effectiveness_rate = (effective / total * 100) if total > 0 else 0
            
            # Calculate pixel change statistics
            pixel_changes = action_pixel_stats[action_id]
            avg_pixels = sum(pixel_changes) / len(pixel_changes) if pixel_changes else 0
            max_pixels = max(pixel_changes) if pixel_changes else 0
            min_pixels = min(pixel_changes) if pixel_changes else 0
            
            print(f"   ACTION{action_id}: {effective}/{total} effective ({effectiveness_rate:.1f}%) | "
                  f"pixels: avg={avg_pixels:.1f}, max={max_pixels}, min={min_pixels}")
        
        # Save detailed results for cross-game analysis
        self._save_results_to_file()
        
        print(f"\n💾 Results saved for cross-game analysis")
        print(f"🔍 Compare with isolated testing results in agents/rl/docs/")

    def _save_results_to_file(self) -> None:
        """Save detailed results to JSON file for cross-game analysis."""
        import json
        import os
        from datetime import datetime
        
        # Ensure directory exists
        results_dir = "agents/rl/docs"
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare results data
        results = {
            "game_id": self.game_id,
            "test_type": "comprehensive_sequential",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.game_start_time,
            "total_actions": len(self.change_log),
            "test_sequence": self.test_sequence,
            "detailed_log": self.change_log,
            "summary": self._generate_summary_stats()
        }
        
        # Save to game-specific file
        filename = f"sequential_test_{self.game_id.replace('-', '_')}.json"
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Detailed results saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    def _generate_summary_stats(self) -> dict:
        """Generate summary statistics for the test."""
        from collections import Counter, defaultdict
        
        if not self.change_log:
            return {}
        
        action_counts = Counter([change["action_id"] for change in self.change_log])
        effective_counts = Counter([change["action_id"] for change in self.change_log if change["pixel_changes"] > 0])
        
        # Generate per-action statistics
        action_stats = {}
        for action_id in sorted(action_counts.keys()):
            total = action_counts[action_id]
            effective = effective_counts.get(action_id, 0)
            effectiveness_rate = (effective / total * 100) if total > 0 else 0
            
            pixel_changes = [change["pixel_changes"] for change in self.change_log if change["action_id"] == action_id]
            avg_pixels = sum(pixel_changes) / len(pixel_changes) if pixel_changes else 0
            
            action_stats[str(action_id)] = {
                "total_tests": total,
                "effective_tests": effective,
                "effectiveness_rate": effectiveness_rate,
                "average_pixel_changes": avg_pixels,
                "is_no_op": effectiveness_rate == 0
            }
        
        return {
            "total_actions_tested": len(self.change_log),
            "overall_effectiveness_rate": sum(1 for c in self.change_log if c["pixel_changes"] > 0) / len(self.change_log) * 100,
            "per_action_stats": action_stats,
            "universal_no_ops": [action_id for action_id, stats in action_stats.items() if stats["is_no_op"]],
            "effective_actions": [action_id for action_id, stats in action_stats.items() if not stats["is_no_op"]]
        }