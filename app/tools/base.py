"""Base interface for all agent tools.

Every tool (EDA, feature engineering, anomaly detection, risk classification,
explanation) implements this interface. This lets the orchestrator invoke
any tool uniformly, and lets each tool be unit tested in isolation without
needing the orchestrator or the LLM.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.schemas import ToolResult


class Tool(ABC):
    """Abstract base class for a single agent-invokable capability."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a uniform ToolResult.

        Implementations should catch their own exceptions and return
        ToolResult(success=False, error=...) rather than raising, so a
        single failing tool cannot crash the orchestrator loop.
        """
        raise NotImplementedError
