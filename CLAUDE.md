# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ARC-AGI-3 agent framework for building and running AI agents that compete in the ARC-AGI-3 competition. The framework provides a structured way to create agents that interact with games via REST API calls to solve visual reasoning puzzles.

## Development Commands

### Environment Setup
- Copy `.env-example` to `.env` and set your `ARC_API_KEY`
- Install dependencies: `uv sync`
- Install with AgentOps observability: `uv sync --agentops`

### Running Agents
- Run an agent: `uv run main.py --agent=AGENT_NAME --game=GAME_ID`
- Run random agent on ls20 game: `uv run main.py --agent=random --game=ls20`
- Run agent with tags: `uv run main.py --agent=random --game=ls20 --tags="experiment,v1.0"`

### Testing and Code Quality
- Run tests: `pytest`
- Run linting: `ruff check`
- Run formatting: `ruff format`
- Run type checking: `mypy`
- Install pre-commit hooks: `pre-commit install` (requires `pip install pre-commit`)

**Note**: The repository contains a `game_loop.md` file that duplicates the README.md content and does not contain specific game loop documentation.

## Architecture

### Core Components

1. **Agent System** (`agents/agent.py`):
   - `Agent` - Abstract base class for all agents with main game loop
   - `Playback` - Special agent that replays recorded sessions
   - Agents must implement `is_done()` and `choose_action()` methods
   - Built-in action counting, timing, and recording capabilities

2. **Game Actions** (`agents/structs.py`):
   - `GameAction` enum defines available actions (RESET, ACTION1-6)
   - Actions support optional reasoning data up to 16KB

3. **Game State Management**:
   - `FrameData` represents current game state with visual frame data
   - `GameState` enum tracks game progression (NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER)
   - `Scorecard` tracks performance across multiple games and plays

4. **Swarm Orchestration** (`agents/swarm.py`):
   - Manages multiple agents playing multiple games concurrently
   - Handles scorecard creation and API communication
   - Provides web interface links for viewing results

5. **Recording System** (`agents/recorder.py`):
   - Automatically records all agent actions and outcomes
   - Stored as JSONL files in `recordings/` directory
   - Recorded sessions can be replayed using the Playback agent

## Game Process

### Visual Puzzle Structure
ARC-AGI-3 games present visual reasoning puzzles as 64x64 grids where each cell contains a color value (integer 0-15). The games involve pattern recognition, spatial reasoning, and logical transformations.

**Example from ls20 game:**
- Game presents a complex visual scene with different colored regions
- Numbers represent colors: 0=black, 3=green, 4=yellow, 5=gray, 8=orange, 9=red, 12=pink, 15=white
- Agents must understand spatial relationships and apply transformations

### Agent Session Flow
1. **Game Selection**: Agent receives list of available games from `/api/games`
2. **Session Initialization**: Creates scorecard via `/api/scorecard/open` 
3. **Game Reset**: First action is always RESET to initialize the game state
4. **Action Loop**: Agent repeatedly:
   - Analyzes current frame data (64x64 grid)
   - Chooses from available actions (RESET, ACTION1-5, ACTION6 with coordinates)
   - Submits action via `/api/cmd/{action_name}`
   - Receives updated frame with new state and score
5. **Termination**: Loop ends when agent wins, hits MAX_ACTIONS, or exits
6. **Cleanup**: Submits final scorecard and saves recording

### Action Types & Game Mechanics
- **Simple Actions (ACTION1-5)**: No parameters, affect the game state globally
- **Complex Action (ACTION6)**: Requires x,y coordinates (0-63), targets specific locations
- **Game States**: NOT_PLAYED → NOT_FINISHED → WIN/GAME_OVER
- **Reasoning Support**: All actions can include up to 16KB of reasoning data
- **Performance Tracking**: System monitors actions/second (FPS), scores, and timing

### Recording Format
Every agent session is automatically recorded as JSONL with:
- Timestamped action sequences
- Complete frame data for each step (visual state as 64x64 arrays)
- Action inputs including coordinates and reasoning
- Game state transitions and scores
- Unique session GUIDs for playback capability

### Common Patterns Observed
- Agents typically run ~80 actions before hitting MAX_ACTIONS limit
- Performance stabilizes around 3-4 actions per second
- Score often remains 0 during exploration phase of random agents
- Frame data shows gradual changes as actions affect the visual puzzle
- Recording files can be substantial (600KB+) due to full visual state capture

## Agents

### Agent Templates

Available in `agents/templates/`:
- `Random` - Makes random valid actions
- `ReasoningAgent` - Base for reasoning-based agents
- `LLM`, `FastLLM`, `GuidedLLM`, `ReasoningLLM` - Various LLM-based agents
- `SmolCodingAgent`, `SmolVisionAgent` - Smolagents integration

### Agent Discovery

Agents are automatically discovered via `agents/__init__.py`:
- All Agent subclasses are registered in `AVAILABLE_AGENTS`
- Recording files are also registered as playback agents
- Use agent class name in lowercase as the agent parameter


### Creating Custom Agents

1. Inherit from `Agent` base class
2. Implement required abstract methods:
   - `is_done(frames, latest_frame)` - Return True when agent should stop
   - `choose_action(frames, latest_frame)` - Return the next GameAction
3. Set `MAX_ACTIONS` class variable to prevent infinite loops
4. Register in `agents/__init__.py` if not auto-discovered

### Agent Session Lifecycle

1. Agent initializes with game_id, card_id, and connection details
2. Main loop runs until `is_done()` returns True or MAX_ACTIONS reached
3. Each iteration calls `choose_action()` and executes the returned action
4. All actions and responses are automatically recorded
5. Cleanup handles scorecard submission and resource cleanup

### API Communication

- All agents communicate with ARC-AGI-3 server via REST API
- Authentication via `ARC_API_KEY` environment variable
- Server URL configurable via `SCHEME`, `HOST`, `PORT` environment variables
- Actions submitted to `/api/cmd/{action_name}` endpoints

### Observability Integration
Read the `Observability` section in the `README.md` for further information
- AgentOps integration for session tracing and analytics
- Sessions automatically tagged with agent type and configuration
- Trace URLs provided in console output for real-time monitoring
- Use `@trace_agent_session` decorator on agent main methods

# Swarms

> Orchestrate agents across multiple games.

Swarms are used to orchestrate your agent across multiple games simultaneously.

Each `swarm`:

* Creates one agent instance per [game](/games)
* Runs all agents concurrently using threads
* Automatically manages [scorecard](/scorecards) opening and closing
* Handles cleanup when all agents complete
* Provides a link to view [replay](/recordings) online

### Running the Agent Swarm

The agent swarm is executed through `main.py`, which manages agent execution across multiple games with automatic scorecard tracking.

### Swarm Command

```bash
uv run main.py --agent <agent_name> [--game <game_filter>] [--tags <tag_list>]
```

### CLI Arguments

| Argument  | Short | Required | Description                                                                                                                                                                                                                                |
| --------- | ----- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--agent` | `-a`  | Yes      | Choose which agent to run. Available agents can be found in the `agents/` directory.                                                                                                                                                       |
| `--game`  | `-g`  | No       | Filter [games](/games) by ID prefix. Can be comma-separated for multiple filters (e.g., `ls20,ft09`). If not specified, the agent plays all available games.                                                                               |
| `--tags`  | `-t`  | No       | Comma-separated list of tags for the scorecard (e.g., `experiment,v1.0`). Tags help categorize and track different agent runs. Helpful when you want to compare different agents. Tags will be recorded on your [scorecards](/scorecards). |

### Examples

```bash
# Run the random agent on all games
uv run main.py --agent=random

# Run an LLM agent on only the ls20 game
uv run main.py --agent=llm --game=ls20

# Run with custom tags for tracking
uv run main.py --agent=llm --tags="experiment,gpt-4,baseline"

# Run against an explicit list of games
uv run main.py --agent=random --game="ls20,ft09"
```



## Testing

Test files located in `tests/unit/`:
- `test_core.py` - Core functionality tests
- `test_recorder.py` - Recording system tests  
- `test_swarm.py` - Swarm orchestration tests

Tests use pytest with async support and request mocking capabilities.