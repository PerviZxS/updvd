# Proposer interface and a deterministic scripted proposer.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# A proposed tool call: tool name plus raw, unvalidated arguments.
@dataclass(frozen=True)
class Proposal:
    tool: str
    args: dict[str, Any]


# Anything that turns a task (and optional rejection feedback) into a proposal.
class Proposer(ABC):
    @abstractmethod
    def propose(self, task: str, feedback: str | None = None) -> Proposal: ...


# Replays a fixed list of proposals per task; repeats the last one once the list is exhausted.
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
