from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable

from pydantic import ValidationError

from .contracts import CONTRACTS
from .engine import Engine, Outcome
from .interceptor import Interceptor
from .schemas import SCHEMAS
from .state import State
from .verdicts import Verdict


@dataclass
class Metrics:
    total_tasks: int = 0
    total_proposals: int = 0
    committed: int = 0
    blocked: int = 0
    recoverable_tasks: int = 0
    recovered_tasks: int = 0
    eval_seconds: list[float] = field(default_factory=list)
    attempts_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        if self.recoverable_tasks == 0:
            return 0.0
        return self.recovered_tasks / self.recoverable_tasks

    @property
    def overhead_ms(self) -> dict[str, float]:
        if not self.eval_seconds:
            return {"p50": 0.0, "p99": 0.0, "mean": 0.0}
        ordered = sorted(self.eval_seconds)
        index99 = min(len(ordered) - 1, int(0.99 * len(ordered)))
        return {
            "p50": statistics.median(ordered) * 1000,
            "p99": ordered[index99] * 1000,
            "mean": statistics.fmean(ordered) * 1000,
        }

    def as_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "total_proposals": self.total_proposals,
            "committed": self.committed,
            "blocked": self.blocked,
            "recoverable_tasks": self.recoverable_tasks,
            "recovered_tasks": self.recovered_tasks,
            "recovery_rate": self.recovery_rate,
            "overhead_ms": self.overhead_ms,
            "attempts_histogram": dict(sorted(self.attempts_histogram.items())),
        }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    # Wilson score interval for a binomial proportion. Preferred over the normal
    # approximation here because the rates are far from 0.5 and the trial counts
    # are small, where the normal interval misbehaves. z = 1.96 is the 95% level.
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, center - half), min(1.0, center + half))


def containment_holds(interceptor: Interceptor, initial_state: State) -> bool:
    # Independently re-derive the run from its starting state. Every committed
    # action in the log must still pass its own contract when replayed here. This
    # re-checks the contracts rather than trusting the verdict the interceptor
    # already wrote, so it can actually fail if a bad action ever slipped through.
    state = initial_state
    for entry in interceptor.trace.entries:
        if not entry.committed:
            continue
        contract = CONTRACTS.get(entry.tool)
        schema = SCHEMAS.get(entry.tool)
        if contract is None or schema is None:
            return False
        try:
            args = schema(**entry.args)
        except ValidationError:
            return False
        if not contract.precondition(state, args).ok:
            return False
        after, produced = contract.apply(state, args)
        if not contract.postcondition(state, after, args, produced).ok:
            return False
        state = after
    return True


def _record(metrics: Metrics, outcome: Outcome) -> None:
    metrics.total_tasks += 1
    for attempt in outcome.attempts:
        metrics.total_proposals += 1
        metrics.eval_seconds.append(attempt.eval_seconds)
        if attempt.decision.committed:
            metrics.committed += 1
        else:
            metrics.blocked += 1

    if outcome.first_verdict is not Verdict.OK:
        metrics.recoverable_tasks += 1
        if outcome.recovered:
            metrics.recovered_tasks += 1

    bucket = len(outcome.attempts) if outcome.committed else -1
    metrics.attempts_histogram[bucket] = metrics.attempts_histogram.get(bucket, 0) + 1


def evaluate(engine: Engine, tasks: Iterable[str]) -> Metrics:
    metrics = Metrics()
    for task in tasks:
        _record(metrics, engine.run(task))
    return metrics