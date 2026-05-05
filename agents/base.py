"""
Base agent template.

Each role copies this file as the starting point for their own agent.
It enforces the basic contract: name, run method, decision logging,
schema validation.

Don't sub-class this if you don't want to. The point is consistency,
not OOP purity.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from agents.decision_log import log_decision


class BaseAgent(ABC):
    """Minimal contract for every agent in the pipeline."""

    name: str = "base_agent"  # Override in subclass

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Each agent implements this. Inputs and outputs are typed via Pydantic schemas."""
        pass

    def log(self, decision_type: str, details: dict, **kwargs):
        """Convenience wrapper around the shared decision logger."""
        log_decision(
            agent=self.name, decision_type=decision_type, details=details, **kwargs
        )
