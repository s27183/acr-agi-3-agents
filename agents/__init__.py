from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .templates.nooptest_agent import NoOpTest
from .templates.comprehensive_nooptest_agent import ComprehensiveNoOpTest

# RL agents status set below based on import success

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

# RL agents
# Training agent is handled specially in main.py
AVAILABLE_AGENTS["rltraining"] = Agent  # Placeholder - handled in main.py

# Import and register RL inference agent
try:
    from .rl.rl_agent import RLAgent
    AVAILABLE_AGENTS["rlagent"] = RLAgent
    RL_AGENTS_AVAILABLE = True
except ImportError:
    # RL dependencies not available, skip RL inference agent
    RL_AGENTS_AVAILABLE = False

# NoOpTest agent imported above

# Base exports
__all__ = [
    "Swarm",
    "Random", 
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "NoOpTest",
    "ComprehensiveNoOpTest", 
    "Agent",
    "Recorder", 
    "Playback",
    "AVAILABLE_AGENTS",
    "RL_AGENTS_AVAILABLE",
]

# RL functionality is in agents.rl module
# RLAgent available if dependencies are installed
