from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Proposal:
    tool: str
    args: dict[str, Any]


class Proposer(ABC):
    @abstractmethod
    def propose(self, task: str, feedback: str | None = None) -> Proposal: ...


class ScriptedProposer(Proposer):
    def __init__(self, script: dict[str, list[Proposal]]) -> None:
        self._script = script
        self._cursor: dict[str, int] = {}

    def propose(self, task: str, feedback: str | None = None) -> Proposal:
        steps = self._script[task]
        index = self._cursor.get(task, 0)
        index = min(index, len(steps) - 1)
        self._cursor[task] = index + 1
        return steps[index]