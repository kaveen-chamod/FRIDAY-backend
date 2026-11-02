from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModule(ABC):
    TOOL: dict[str, Any]

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        raise NotImplementedError
