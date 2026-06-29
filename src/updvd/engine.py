from __future__ import annotations

import time
from dataclasses import dataclass

from .interceptor import Decision, Interceptor
from .proposer import Proposer
from .verdicts import Verdict


@dataclass(frozen=True)
class Attempt:
    decision: Decision
    eval_seconds: float


@dataclass(frozen=True)
class Outcome:
    task: str
    attempts: list[Attempt]

    @property
    def committed(self) -> bool:
        return self.attempts[-1].decision.committed

    @property
    def recovered(self) -> bool:
        return len(self.attempts) > 1 and self.committed

    @property
    def first_verdict(self) -> Verdict:
        return self.attempts[0].decision.verdict


class Engine:
    # Drives one task: ask the proposer for a call, let the interceptor judge it,
    # and on a rejection feed the error back so the proposer can try again.
    def __init__(self, interceptor: Interceptor, proposer: Proposer, max_retries: int = 3) -> None:
        self._interceptor = interceptor
        self._proposer = proposer
        self._max_retries = max_retries

    def run(self, task: str) -> Outcome:
        attempts: list[Attempt] = []
        feedback: str | None = None

        for _ in range(self._max_retries + 1):
            proposal = self._proposer.propose(task, feedback)
            # Time only the check, not the model, so overhead reflects the layer.
            start = time.perf_counter()
            decision = self._interceptor.propose(proposal.tool, proposal.args)
            elapsed = time.perf_counter() - start
            attempts.append(Attempt(decision, elapsed))

            if decision.committed:
                break
            feedback = decision.error_signal

        return Outcome(task, attempts)