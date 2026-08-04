"""Module1 multi-step task environment.

This package consumes the Module0 KataGo feature store and exposes a small
Gym-style environment for downstream Module2/3/4 experiments.
"""

from .adapters.module0_adapter import Module0Adapter
from .core.environment import MultiStepTaskEnv
from .core.go_state import GoState
from .core.transition import TransitionRecord
from .datasets.episode_manifest import EpisodeManifest, EpisodeSpec

__all__ = [
    "EpisodeManifest",
    "EpisodeSpec",
    "GoState",
    "Module0Adapter",
    "MultiStepTaskEnv",
    "TransitionRecord",
]
