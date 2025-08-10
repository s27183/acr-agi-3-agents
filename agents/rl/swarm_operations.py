"""
Swarm Operations Wrapper for Gymnasium Environment

This module provides a clean interface for interacting with the ARC-AGI-3 API
through swarm operations, designed to be used by Gymnasium environments.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any, Tuple
from ..structs import FrameData, GameAction, GameState, SimpleAction, ComplexAction

logger = logging.getLogger(__name__)


class SwarmOperations:
    """
    Handles all swarm/agent/API operations for a single game instance.
    
    This wrapper encapsulates:
    - API connection management
    - Game ID resolution
    - Scorecard lifecycle
    - Action execution
    - Error handling
    """
    
    def __init__(self, 
                 game_id: str, 
                 tags: Optional[list] = None,
                 scorecard_id: Optional[str] = None,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize swarm operations for a game.
        
        Args:
            game_id: Game identifier (can be short form like "ls20")
            tags: Optional tags for scorecard tracking
            scorecard_id: Optional existing scorecard ID to use
            root_url: Optional API root URL
            api_key: Optional API key
        """
        self.requested_game_id = game_id
        self.tags = tags or ["rl_training", "gymnasium_env"]
        
        # Use provided values or fall back to environment
        if root_url:
            self.root_url = root_url
        else:
            self.scheme = os.environ.get("SCHEME", "http")
            self.host = os.environ.get("HOST", "localhost") 
            self.port = os.environ.get("PORT", 8001)
            self.root_url = f"{self.scheme}://{self.host}:{self.port}"
        
        # Use provided API key or fall back to environment
        api_key_to_use = api_key if api_key is not None else os.getenv("ARC_API_KEY", "")
        self.headers = {
            "X-API-Key": api_key_to_use,
            "Accept": "application/json",
        }
        
        # Create session for connection reuse
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Game state
        self.guid = ""
        self.card_id = scorecard_id  # Use provided scorecard if available
        self.resolved_game_id = None
        self.last_frame = None
        self._created_own_scorecard = False  # Track if we created our own scorecard
        
        # Resolve game ID and optionally create scorecard
        self._initialize_game()
    
    def _initialize_game(self) -> None:
        """Initialize game by resolving ID and creating scorecard."""
        # Resolve game ID
        self.resolved_game_id = self._resolve_game_id(self.requested_game_id)
        logger.info(f"Resolved game ID: {self.requested_game_id} -> {self.resolved_game_id}")
        
        # Create scorecard only if one wasn't provided
        if self.card_id is None:
            self._create_scorecard()
            self._created_own_scorecard = True
        else:
            logger.info(f"Using provided scorecard: {self.card_id}")
            self._created_own_scorecard = False
    
    def _resolve_game_id(self, game_id: str) -> str:
        """
        Resolve short game ID to full ID via API.
        
        Args:
            game_id: Short or full game ID
            
        Returns:
            Full resolved game ID
        """
        try:
            # Get available games
            response = self._session.get(
                f"{self.root_url}/api/games",
                timeout=30
            )
            
            if response.status_code == 200:
                games_data = response.json()
                
                # Extract game IDs from API response (API returns list of dicts with game_id keys)
                games = [g["game_id"] for g in games_data] if games_data else []
                
                # Find matching game
                for game in games:
                    if game.startswith(game_id):
                        return game
                
                # If no match found, return original
                logger.warning(f"No game found matching '{game_id}', using as-is")
                return game_id
            else:
                logger.error(f"Failed to get games list: {response.status_code}")
                return game_id
                
        except Exception as e:
            logger.error(f"Error resolving game ID: {e}")
            return game_id
    
    def _create_scorecard(self) -> None:
        """Create a scorecard for this session."""
        try:
            response = self._session.post(
                f"{self.root_url}/api/scorecard/open",
                json={"tags": self.tags},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.card_id = data.get("card_id")
                logger.info(f"Created scorecard: {self.card_id}")
            else:
                logger.warning(f"Failed to create scorecard: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not create scorecard: {e}")
    
    def reset_game(self) -> Optional[FrameData]:
        """
        Reset the game and return initial frame.
        
        Returns:
            Initial frame data or None on error
        """
        try:
            data = {
                "game_id": self.resolved_game_id,
            }
            if self.card_id:
                data["card_id"] = self.card_id
            if self.guid:
                data["guid"] = self.guid
            
            response = self._session.post(
                f"{self.root_url}/api/cmd/RESET",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                frame_data = response.json()
                
                if "error" in frame_data:
                    logger.error(f"API error during reset: {frame_data['error']}")
                    return None
                
                frame = FrameData.model_validate(frame_data)
                if frame.guid:
                    self.guid = frame.guid
                    
                self.last_frame = frame
                return frame
            else:
                logger.error(f"Reset failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Reset game failed: {e}")
            return None
    
    def execute_action(self, action_id: int, x: Optional[int] = None, y: Optional[int] = None) -> Optional[FrameData]:
        """
        Execute an action and return resulting frame.
        
        Args:
            action_id: Action ID (0-5 for RESET, ACTION1-5)
            x: Optional x coordinate for ACTION6
            y: Optional y coordinate for ACTION6
            
        Returns:
            Resulting frame data or None on error
        """
        # Map action ID to GameAction
        action = self._map_action(action_id, x, y)
        
        try:
            data = action.action_data.model_dump()
            if self.guid:
                data["guid"] = self.guid
            if self.resolved_game_id:
                data["game_id"] = self.resolved_game_id
            
            response = self._session.post(
                f"{self.root_url}/api/cmd/{action.name}",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                frame_data = response.json()
                
                if "error" in frame_data:
                    logger.error(f"API error during action {action.name}: {frame_data['error']}")
                    return None
                
                frame = FrameData.model_validate(frame_data)
                if frame.guid:
                    self.guid = frame.guid
                    
                self.last_frame = frame
                return frame
            else:
                logger.error(f"Action {action.name} failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Execute action failed: {e}")
            return None
    
    def _map_action(self, action_id: int, x: Optional[int] = None, y: Optional[int] = None) -> GameAction:
        """Map action ID to GameAction enum."""
        action_map = {
            0: GameAction.RESET,
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6
        }
        
        game_action = action_map.get(action_id, GameAction.RESET)
        
        # Handle ACTION6 with coordinates
        if game_action == GameAction.ACTION6 and x is not None and y is not None:
            game_action.action_data = ComplexAction(x=x, y=y)
        
        return game_action
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current game state information.
        
        Returns:
            Dictionary with state information
        """
        if self.last_frame:
            return {
                "score": self.last_frame.score,
                "state": self.last_frame.state.value,
                "guid": self.guid,
                "card_id": self.card_id,
                "game_id": self.resolved_game_id
            }
        else:
            return {
                "score": 0,
                "state": "NOT_PLAYED",
                "guid": self.guid,
                "card_id": self.card_id,
                "game_id": self.resolved_game_id
            }
    
    def query_game_capabilities(self) -> Dict[str, Any]:
        """
        Query the game's supported actions and capabilities.
        
        Returns:
            Dictionary with game capabilities including supported actions
        """

        return self._infer_game_capabilities()

    
    def _infer_game_capabilities(self) -> Dict[str, Any]:
        """
        Infer game capabilities by testing available actions.
        
        Returns:
            Dictionary with inferred capabilities
        """
        supported_actions = [0]  # RESET is always supported
        
        # Test each action by attempting to execute it after a reset
        try:
            # First reset the game to a known state
            reset_frame = self.reset_game()
            if reset_frame is None:
                logger.error("Could not reset game for capability testing")
                return {"supported_actions": [0, 1, 2, 3, 4], "max_action": 4}  # Conservative fallback
            
            # Test actions 1-6 with very short timeout
            for action_id in range(1, 7):
                try:
                    # Store current state
                    original_guid = self.guid
                    
                    # Try the action with short timeout
                    # For ACTION6, provide sample coordinates (center of grid)
                    if action_id == 6:
                        action = self._map_action(action_id, x=32, y=32)
                    else:
                        action = self._map_action(action_id)
                    
                    data = action.action_data.model_dump()
                    if self.guid:
                        data["guid"] = self.guid
                    if self.resolved_game_id:
                        data["game_id"] = self.resolved_game_id
                    
                    response = self._session.post(
                        f"{self.root_url}/api/cmd/{action.name}",
                        json=data,
                        timeout=5  # Very short timeout for testing
                    )
                    
                    if response.status_code == 200:
                        frame_data = response.json()
                        if "error" not in frame_data:
                            supported_actions.append(action_id)
                            logger.debug(f"Action {action_id} supported")
                        else:
                            logger.debug(f"Action {action_id} returned error: {frame_data.get('error')}")
                    else:
                        logger.debug(f"Action {action_id} failed with status {response.status_code}")
                        
                except Exception as e:
                    logger.debug(f"Action {action_id} failed during testing: {e}")
                    # Don't add to supported actions
            
            # Reset game again to clean state
            self.reset_game()
            
        except Exception as e:
            logger.error(f"Capability inference failed: {e}")
            # Conservative fallback - assume only basic actions work
            supported_actions = [0, 1, 2, 3, 4]
        
        max_action = max(supported_actions) if supported_actions else 4
        
        logger.info(f"Game {self.resolved_game_id} supports actions: {supported_actions} (max: {max_action})")
        
        # Note: ACTION6 requires x,y coordinates and cannot be used directly with Discrete action spaces
        # RL agents using Discrete action spaces should exclude ACTION6 from their action space
        if 6 in supported_actions:
            logger.info(f"Note: ACTION6 requires coordinates (x,y) and is not compatible with Discrete action spaces")
        
        return {
            "supported_actions": supported_actions,
            "max_action": max_action,
            "total_actions": len(supported_actions),
            "inferred": True
        }
    
    def close_scorecard(self) -> None:
        """Close the scorecard for this session."""
        # Only close scorecard if we created it ourselves
        # Don't close if it was provided externally (the swarm will handle it)
        if self._created_own_scorecard and self.card_id:
            logger.info(f"Closing self-created scorecard: {self.card_id}")
            # Note: In training, environments typically finish before explicit closing is needed
            # The server may auto-cleanup inactive scorecards, but we track this for completeness
        else:
            logger.debug(f"Not closing external scorecard: {self.card_id}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.close_scorecard()
        
        if hasattr(self, '_session'):
            self._session.close()